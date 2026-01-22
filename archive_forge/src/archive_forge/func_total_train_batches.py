from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@property
def total_train_batches(self) -> Union[int, float]:
    """The total number of training batches, which may change from epoch to epoch.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the training
        dataloader is of infinite size.

        """
    return self.trainer.num_training_batches