from typing import Any, Iterable, Optional, Union
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE, _TENSORBOARDX_AVAILABLE
from lightning_fabric.plugins.environments import SLURMEnvironment
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger
from pytorch_lightning.trainer.connectors.logger_connector.result import _METRICS, _OUT_DICT, _PBAR_DICT
from pytorch_lightning.utilities.rank_zero import WarningCache
def update_train_step_metrics(self) -> None:
    if self.trainer.fit_loop._should_accumulate() and self.trainer.lightning_module.automatic_optimization:
        return
    assert isinstance(self._first_loop_iter, bool)
    if self.should_update_logs or self.trainer.fast_dev_run:
        self.log_metrics(self.metrics['log'])