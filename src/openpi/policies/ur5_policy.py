"""UR5e policy transforms for pi0 fine-tuning.

Our dataset has:
  - joint_position:   float32 (6,)   — 6 arm joints
  - gripper_position: float32 (1,)   — absolute gripper
  - actions:          float32 (7,)   — delta joints (6) + abs gripper (1)

pi0_base was pretrained on 8D state (7 arm joints + 1 gripper, DROID convention).
We pad our 6-joint state with one zero to reach 7, then concat gripper → 8D.
Actions are padded to 8D during training and sliced back to 7D at inference.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5_example() -> dict:
    """Creates a random input example for the UR5 policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(6).astype(np.float32),
        "observation/gripper_position": np.random.rand(1).astype(np.float32),
        "prompt": "lift the arm up",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Ur5Inputs(transforms.DataTransformFn):
    """Converts UR5e observations to pi0 model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]

        # UR5e has 6 joints; pad to 7 with a zero to match pi0_base's 8D state
        # (pretrained on DROID which has 7 arm joints + 1 gripper = 8D)
        joint_pos = np.asarray(data["observation/joint_position"])
        joint_pos_padded = np.concatenate([joint_pos, np.zeros(1, dtype=joint_pos.dtype)])
        state = np.concatenate([joint_pos_padded, gripper_pos])  # (8,)

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            # Pad 7D actions → 8D to match model output dimension
            if actions.ndim == 2 and actions.shape[1] == 7:
                pad = np.zeros((actions.shape[0], 1), dtype=actions.dtype)
                actions = np.concatenate([actions, pad], axis=1)
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Ur5Outputs(transforms.DataTransformFn):
    """Converts pi0 model outputs back to UR5e actions."""

    def __call__(self, data: dict) -> dict:
        # Model outputs 8D; slice to 7D (6 joint deltas + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :7])}
