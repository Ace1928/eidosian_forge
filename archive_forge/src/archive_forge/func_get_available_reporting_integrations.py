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
def get_available_reporting_integrations():
    integrations = []
    if is_azureml_available() and (not is_mlflow_available()):
        integrations.append('azure_ml')
    if is_comet_available():
        integrations.append('comet_ml')
    if is_dagshub_available():
        integrations.append('dagshub')
    if is_dvclive_available():
        integrations.append('dvclive')
    if is_mlflow_available():
        integrations.append('mlflow')
    if is_neptune_available():
        integrations.append('neptune')
    if is_tensorboard_available():
        integrations.append('tensorboard')
    if is_wandb_available():
        integrations.append('wandb')
    if is_codecarbon_available():
        integrations.append('codecarbon')
    if is_clearml_available():
        integrations.append('clearml')
    return integrations