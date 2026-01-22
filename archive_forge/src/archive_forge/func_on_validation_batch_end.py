import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
@override
def on_validation_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
    if self.is_disabled:
        return
    if trainer.sanity_checking:
        self._update(self.val_sanity_progress_bar_id, batch_idx + 1)
    elif self.val_progress_bar_id is not None:
        self._update(self.val_progress_bar_id, batch_idx + 1)
    self.refresh()