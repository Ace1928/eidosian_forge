import logging
import os
import tempfile
import warnings
from packaging.version import Version
import mlflow.pytorch
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.pytorch import _pytorch_autolog
from mlflow.utils.autologging_utils import (
from mlflow.utils.checkpoint_utils import MlflowModelCheckpointCallbackBase
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
@rank_zero_only
def on_validation_epoch_end(self, trainer, pl_module):
    """
            Log loss and other metrics values after each validation epoch

            Args:
                trainer: pytorch lightning trainer instance
                pl_module: pytorch lightning base module
            """
    self._log_epoch_metrics(trainer, pl_module)