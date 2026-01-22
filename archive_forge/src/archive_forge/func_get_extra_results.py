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
def get_extra_results(self, trainer: 'pl.Trainer') -> Dict[str, Any]:
    """Gather extra state from the Trainer and return it as a dictionary for sending back to the main process. To
        avoid issues with memory sharing, we cast the data to numpy.

        Args:
            trainer: reference to the Trainer.

        Returns:
            A dictionary with items to send back to the main process where :meth:`update_main_process_results` will
            process this output.

        """
    callback_metrics: dict = apply_to_collection(trainer.callback_metrics, Tensor, lambda x: x.cpu().numpy())
    return {'callback_metrics': callback_metrics}