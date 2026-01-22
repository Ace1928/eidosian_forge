import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402

        Setup the optional DVCLive integration. To customize this callback beyond the environment variables below, see
        [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).

        Environment:
        - **HF_DVCLIVE_LOG_MODEL** (`str`, *optional*):
            Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [`Trainer`]. If set to `True` or
            *1*, the final checkpoint is logged at the end of training. If set to `all`, the entire
            [`TrainingArguments`]'s `output_dir` is logged at each checkpoint.
        