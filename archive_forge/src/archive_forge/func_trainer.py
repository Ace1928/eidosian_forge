from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@property
def trainer(self) -> 'pl.Trainer':
    if self._trainer is None:
        raise TypeError(f'The `{self.__class__.__name__}._trainer` reference has not been set yet.')
    return self._trainer