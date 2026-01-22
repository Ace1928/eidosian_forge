import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
@property
def parallel_mode(self):
    """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - `ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - `ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses `torch.nn.DataParallel`).
        - `ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          `torch.nn.DistributedDataParallel`).
        - `ParallelMode.TPU`: several TPU cores.
        """
    requires_backends(self, ['torch'])
    if is_torch_tpu_available():
        return ParallelMode.TPU
    elif is_sagemaker_mp_enabled():
        return ParallelMode.SAGEMAKER_MODEL_PARALLEL
    elif is_sagemaker_dp_enabled():
        return ParallelMode.SAGEMAKER_DATA_PARALLEL
    elif self.distributed_state is not None and self.distributed_state.distributed_type != DistributedType.NO or (self.distributed_state is None and self.local_rank != -1):
        return ParallelMode.DISTRIBUTED
    elif self.n_gpu > 1:
        return ParallelMode.NOT_DISTRIBUTED
    else:
        return ParallelMode.NOT_PARALLEL