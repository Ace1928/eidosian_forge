from typing import Any, Dict, Type
from torch import Tensor
from torch.optim import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
def on_before_backward(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', loss: Tensor) -> None:
    """Called before ``loss.backward()``."""