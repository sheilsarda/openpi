"""Convert UR5e ROS2 bag files to LeRobot dataset format.

Each bag directory under --bags-dir is treated as one episode.
Actions are computed as delta joint positions (6D) + absolute gripper position (1D).

Usage:
    uv run examples/ur5/convert_ur5_bag_to_lerobot.py \
        --bags-dir ~/Development/VLA_Arm_Controller/training_data \
        --repo-id sheilsarda/ur5_isaac_sim_v1 \
        --task "move the arm up and down"
"""

import dataclasses
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
import tyro


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
IMAGE_SHAPE = (240, 320, 3)  # H x W x C


@dataclasses.dataclass
class Args:
    bags_dir: Path
    repo_id: str
    task: str = "move the arm up and down"
    output_fps: int = 10
    push_to_hub: bool = False


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _decode_image(msg) -> np.ndarray:
    """Decode a ROS2 sensor_msgs/Image to a HxWx3 uint8 RGB array."""
    channels = len(msg.data) // (msg.height * msg.width)
    img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, channels)
    enc = msg.encoding.lower()
    if enc in ("bgr8", "bgr"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif enc in ("bgra8", "bgra"):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif enc in ("rgba8", "rgba"):
        img = img[..., :3][..., ::-1]
    # rgb8 / mono8 pass through; take first 3 channels in any other case
    return img[..., :3].copy()


def _resize(img: np.ndarray) -> np.ndarray:
    h, w = IMAGE_SHAPE[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------

TOPICS = ("/camera/image_raw", "/camera_wrist/image_raw", "/joint_states")


def _read_bag(bag_path: Path, typestore) -> dict[str, list]:
    """Return {topic: [(timestamp_ns, msg), ...]} for the three required topics."""
    data: dict[str, list] = {t: [] for t in TOPICS}
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic in data:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                data[connection.topic].append((timestamp, msg))
    return data


def _nearest(messages: list, target_ns: int):
    """Return the message whose timestamp is closest to target_ns."""
    timestamps = np.array([ts for ts, _ in messages])
    idx = int(np.argmin(np.abs(timestamps - target_ns)))
    return messages[idx][1]


def _joint_positions(msg, names: list[str]) -> np.ndarray:
    name_to_pos = dict(zip(msg.name, msg.position))
    return np.array([name_to_pos.get(j, 0.0) for j in names], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-bag conversion
# ---------------------------------------------------------------------------

def _bag_to_frames(bag_path: Path, typestore, fps: int) -> list[dict] | None:
    data = _read_bag(bag_path, typestore)

    if not all(len(v) >= 2 for v in data.values()):
        missing = [t for t, v in data.items() if len(v) < 2]
        print(f"  Skipping {bag_path.name}: insufficient messages on {missing}")
        return None

    # Clamp to the time window covered by all three topics
    t_start = max(msgs[0][0] for msgs in data.values())
    t_end   = min(msgs[-1][0] for msgs in data.values())
    if t_end <= t_start:
        print(f"  Skipping {bag_path.name}: topics don't overlap in time")
        return None

    period_ns = int(1e9 / fps)
    target_times = range(t_start, t_end, period_ns)

    raw_frames = []
    for t in target_times:
        base_msg  = _nearest(data["/camera/image_raw"],       t)
        wrist_msg = _nearest(data["/camera_wrist/image_raw"], t)
        js_msg    = _nearest(data["/joint_states"],           t)

        raw_frames.append({
            "exterior_image_1_left": _resize(_decode_image(base_msg)),
            "wrist_image_left":      _resize(_decode_image(wrist_msg)),
            "joint_position":        _joint_positions(js_msg, ARM_JOINTS),
            "gripper_position":      _joint_positions(js_msg, [GRIPPER_JOINT]),
        })

    # Actions = delta joints (t→t+1) + absolute gripper at t+1; drop last frame
    frames = []
    for i in range(len(raw_frames) - 1):
        delta   = raw_frames[i + 1]["joint_position"] - raw_frames[i]["joint_position"]
        gripper = raw_frames[i + 1]["gripper_position"]
        frames.append({
            **raw_frames[i],
            "actions": np.concatenate([delta, gripper]).astype(np.float32),
        })

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    typestore = get_typestore(Stores.LATEST)

    bag_dirs = sorted(p for p in args.bags_dir.iterdir() if p.is_dir())
    if not bag_dirs:
        raise ValueError(f"No bag directories found in {args.bags_dir}")
    print(f"Found {len(bag_dirs)} bag(s) in {args.bags_dir}")

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="ur5e",
        fps=args.output_fps,
        features={
            "exterior_image_1_left": {
                "dtype": "video",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "video",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    converted = 0
    for bag_dir in tqdm(bag_dirs, desc="Converting"):
        frames = _bag_to_frames(bag_dir, typestore, args.output_fps)
        if not frames:
            continue
        for frame in frames:
            dataset.add_frame({**frame, "task": args.task})
        dataset.save_episode()
        converted += 1

    print(f"\nDone. Converted {converted}/{len(bag_dirs)} bags.")
    print(f"  Episodes : {dataset.num_episodes}")
    print(f"  Frames   : {dataset.num_frames}")
    print(f"  Saved to : {output_path}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["ur5e", "isaac-sim"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
