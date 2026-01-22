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
def set_logging(self, strategy: Union[str, IntervalStrategy]='steps', steps: int=500, report_to: Union[str, List[str]]='none', level: str='passive', first_step: bool=False, nan_inf_filter: bool=False, on_each_node: bool=False, replica_level: str='passive'):
    """
        A method that regroups all arguments linked to logging.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The logging strategy to adopt during training. Possible values are:

                    - `"no"`: No logging is done during training.
                    - `"epoch"`: Logging is done at the end of each epoch.
                    - `"steps"`: Logging is done every `logging_steps`.

            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two logs if `strategy="steps"`.
            level (`str`, *optional*, defaults to `"passive"`):
                Logger log level to use on the main process. Possible choices are the log levels as strings: `"debug"`,
                `"info"`, `"warning"`, `"error"` and `"critical"`, plus a `"passive"` level which doesn't set anything
                and lets the application set the level.
            report_to (`str` or `List[str]`, *optional*, defaults to `"all"`):
                The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
                `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`,
                `"neptune"`, `"tensorboard"`, and `"wandb"`. Use `"all"` to report to all integrations installed,
                `"none"` for no integrations.
            first_step (`bool`, *optional*, defaults to `False`):
                Whether to log and evaluate the first `global_step` or not.
            nan_inf_filter (`bool`, *optional*, defaults to `True`):
                Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is
                `nan` or `inf` is filtered and the average loss of the current logging window is taken instead.

                <Tip>

                `nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
                gradient is computed or applied to the model.

                </Tip>

            on_each_node (`bool`, *optional*, defaults to `True`):
                In multinode distributed training, whether to log using `log_level` once per node, or only on the main
                node.
            replica_level (`str`, *optional*, defaults to `"passive"`):
                Logger log level to use on replicas. Same choices as `log_level`

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_logging(strategy="steps", steps=100)
        >>> args.logging_steps
        100
        ```
        """
    self.logging_strategy = IntervalStrategy(strategy)
    if self.logging_strategy == IntervalStrategy.STEPS and steps == 0:
        raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
    self.logging_steps = steps
    self.report_to = report_to
    self.log_level = level
    self.logging_first_step = first_step
    self.logging_nan_inf_filter = nan_inf_filter
    self.log_on_each_node = on_each_node
    self.log_level_replica = replica_level
    return self