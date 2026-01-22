from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@property
def total_val_batches(self) -> Union[int, float]:
    """The total number of validation batches, which may change from epoch to epoch for all val dataloaders.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the predict dataloader
        is of infinite size.

        """
    if not self.trainer.fit_loop.epoch_loop._should_check_val_epoch():
        return 0
    return sum(self.trainer.num_val_batches) if isinstance(self.trainer.num_val_batches, list) else self.trainer.num_val_batches