from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@property
def total_test_batches_current_dataloader(self) -> Union[int, float]:
    """The total number of testing batches, which may change from epoch to epoch for current dataloader.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the test dataloader is
        of infinite size.

        """
    batches = self.trainer.num_test_batches
    if isinstance(batches, list):
        assert self._current_eval_dataloader_idx is not None
        return batches[self._current_eval_dataloader_idx]
    return batches