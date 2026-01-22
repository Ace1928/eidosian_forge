import logging
import numbers
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import (
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import ScriptModule, Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from typing_extensions import Self, override
import lightning_fabric as lf
import pytorch_lightning as pl
from lightning_fabric.loggers import Logger as FabricLogger
from lightning_fabric.utilities.apply_func import convert_to_tensors
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_fabric.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning_fabric.wrappers import _FabricOptimizer
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.saving import _load_from_checkpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import _FxValidator
from pytorch_lightning.trainer.connectors.logger_connector.result import _get_default_dtype
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_9_1
from pytorch_lightning.utilities.model_helpers import _restricted_classmethod
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_debug, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import (
def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: Optional[Union[int, float]]=None, gradient_clip_algorithm: Optional[str]=None) -> None:
    """Perform gradient clipping for the optimizer parameters. Called before :meth:`optimizer_step`.

        Args:
            optimizer: Current optimizer being used.
            gradient_clip_val: The value at which to clip gradients. By default, value passed in Trainer
                will be available here.
            gradient_clip_algorithm: The gradient clipping algorithm to use. By default, value
                passed in Trainer will be available here.

        Example::

            def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
                # Implement your own custom logic to clip gradients
                # You can call `self.clip_gradients` with your settings:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm=gradient_clip_algorithm
                )

        """
    self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)