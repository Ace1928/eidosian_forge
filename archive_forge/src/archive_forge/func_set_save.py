import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
def set_save(self, strategy: Union[str, IntervalStrategy]='steps', steps: int=500, total_limit: Optional[int]=None, on_each_node: bool=False):
    """
        A method that regroups all arguments linked to checkpoint saving.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The checkpoint save strategy to adopt during training. Possible values are:

                    - `"no"`: No save is done during training.
                    - `"epoch"`: Save is done at the end of each epoch.
                    - `"steps"`: Save is done every `save_steps`.

            steps (`int`, *optional*, defaults to 500):
                Number of updates steps before two checkpoint saves if `strategy="steps"`.
            total_limit (`int`, *optional*):
                If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
                `output_dir`.
            on_each_node (`bool`, *optional*, defaults to `False`):
                When doing multi-node distributed training, whether to save models and checkpoints on each node, or
                only on the main one.

                This should not be activated when the different nodes use the same storage as the files will be saved
                with the same names for each node.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_save(strategy="steps", steps=100)
        >>> args.save_steps
        100
        ```
        """
    self.save_strategy = IntervalStrategy(strategy)
    if self.save_strategy == IntervalStrategy.STEPS and steps == 0:
        raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
    self.save_steps = steps
    self.save_total_limit = total_limit
    self.save_on_each_node = on_each_node
    return self