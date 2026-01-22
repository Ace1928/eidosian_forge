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
@property
def should_update_logs(self) -> bool:
    trainer = self.trainer
    if trainer.log_every_n_steps == 0:
        return False
    if (loop := trainer._active_loop) is None:
        return True
    if isinstance(loop, pl.loops._FitLoop):
        step = loop.epoch_loop._batches_that_stepped + 1
    elif isinstance(loop, (pl.loops._EvaluationLoop, pl.loops._PredictionLoop)):
        step = loop.batch_progress.current.ready
    else:
        raise NotImplementedError(loop)
    should_log = step % trainer.log_every_n_steps == 0
    return should_log or trainer.should_stop