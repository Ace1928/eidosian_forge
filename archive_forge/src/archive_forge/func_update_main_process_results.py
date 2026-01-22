import logging
import os
import queue
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Union
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.strategies.launchers.multiprocessing import (
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.distributed import _set_num_threads_if_needed
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.strategies.launchers.launcher import _Launcher
from pytorch_lightning.trainer.connectors.signal_connector import _SIGNUM
from pytorch_lightning.trainer.states import TrainerFn, TrainerState
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
def update_main_process_results(self, trainer: 'pl.Trainer', extra: Dict[str, Any]) -> None:
    """Retrieve the :attr:`trainer.callback_metrics` dictionary from the given queue. To preserve consistency, we
        cast back the data to ``torch.Tensor``.

        Args:
            trainer: reference to the Trainer.
            extra: A dictionary with trainer state that was sent from the worker process and needs to be restored
                on the current trainer.

        """
    callback_metrics = extra['callback_metrics']
    trainer.callback_metrics.update(apply_to_collection(callback_metrics, np.ndarray, lambda x: torch.tensor(x)))