from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Literal, Optional, Union
import torch
from torch import Tensor
from torch.optim import LBFGS, Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.accelerators.cuda import _patch_cuda_is_available
from lightning_fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning_fabric.utilities.types import Optimizable
from pytorch_lightning.plugins.precision.precision import Precision
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
def autocast_context_manager(self) -> torch.autocast:
    return torch.autocast(self.device, dtype=torch.bfloat16 if self.precision == 'bf16-mixed' else torch.half)