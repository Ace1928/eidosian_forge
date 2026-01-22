import logging
import math
import os
import warnings
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
from weakref import proxy
import torch
from torch.optim import Optimizer
import pytorch_lightning as pl
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning_fabric.utilities.cloud_io import _is_local_file_protocol
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback, Checkpoint, EarlyStopping, ProgressBar
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.utilities import _log_hyperparams
from pytorch_lightning.loops import _PredictionLoop, _TrainingEpochLoop
from pytorch_lightning.loops.evaluation_loop import _EvaluationLoop
from pytorch_lightning.loops.fit_loop import _FitLoop
from pytorch_lightning.loops.utilities import _parse_loop_limits, _reset_progress
from pytorch_lightning.plugins import _PLUGIN_INPUT, Precision
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import ParallelStrategy, Strategy
from pytorch_lightning.trainer import call, setup
from pytorch_lightning.trainer.configuration_validator import _verify_loop_configurations
from pytorch_lightning.trainer.connectors.accelerator_connector import (
from pytorch_lightning.trainer.connectors.callback_connector import _CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import _DataConnector
from pytorch_lightning.trainer.connectors.logger_connector import _LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, _PBAR_DICT, _ResultCollection
from pytorch_lightning.trainer.connectors.signal_connector import _SignalConnector
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.utilities import GradClipAlgorithmType, parsing
from pytorch_lightning.utilities.argparse import _defaults_from_env_vars
from pytorch_lightning.utilities.compile import _maybe_unwrap_optimized, _verify_strategy_supports_compile
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.utilities.types import (
from pytorch_lightning.utilities.warnings import PossibleUserWarning
@property
def num_sanity_val_batches(self) -> List[Union[int, float]]:
    """The number of validation batches that will be used during the sanity-checking part of ``trainer.fit()``."""
    max_batches = self.fit_loop.epoch_loop.val_loop.max_batches
    return [min(self.num_sanity_val_steps, batches) for batches in max_batches]