import logging
import os
import re
import shutil
import time
import warnings
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set
from weakref import proxy
import torch
import yaml
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import _is_dir, _is_local_file_protocol, get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT
def check_monitor_top_k(self, trainer: 'pl.Trainer', current: Optional[Tensor]=None) -> bool:
    if current is None:
        return False
    if self.save_top_k == -1:
        return True
    less_than_k_models = len(self.best_k_models) < self.save_top_k
    if less_than_k_models:
        return True
    monitor_op = {'min': torch.lt, 'max': torch.gt}[self.mode]
    should_update_best_and_save = monitor_op(current, self.best_k_models[self.kth_best_model_path])
    should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))
    return should_update_best_and_save