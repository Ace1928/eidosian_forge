import importlib
import logging
import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRScheduler, LRSchedulerConfig
def suggestion(self, skip_begin: int=10, skip_end: int=1) -> Optional[float]:
    """This will propose a suggestion for an initial learning rate based on the point with the steepest negative
        gradient.

        Args:
            skip_begin: how many samples to skip in the beginning; helps to avoid too naive estimates
            skip_end: how many samples to skip in the end; helps to avoid too optimistic estimates

        Returns:
            The suggested initial learning rate to use, or `None` if a suggestion is not possible due to too few
            loss samples.

        """
    losses = torch.tensor(self.results['loss'][skip_begin:-skip_end])
    losses = losses[torch.isfinite(losses)]
    if len(losses) < 2:
        log.error('Failed to compute suggestion for learning rate because there are not enough points. Increase the loop iteration limits or the size of your dataset/dataloader.')
        self._optimal_idx = None
        return None
    gradients = torch.gradient(losses)[0]
    min_grad = torch.argmin(gradients).item()
    self._optimal_idx = min_grad + skip_begin
    return self.results['lr'][self._optimal_idx]