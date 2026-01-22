import functools
import inspect
import os
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sized, Tuple, Type, Union
from lightning_utilities.core.inheritance import get_all_subclasses
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, Sampler
from typing_extensions import TypeGuard
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.seed import pl_worker_init_function
def suggested_max_num_workers(local_world_size: int) -> int:
    """Suggests an upper bound of ``num_workers`` to use in a PyTorch :class:`~torch.utils.data.DataLoader` based on
    the number of CPU cores available on the system and the number of distributed processes in the current machine.

    Args:
        local_world_size: The number of distributed processes running on the current machine. Set this to the number
            of devices configured in Fabric/Trainer.

    """
    if local_world_size < 1:
        raise ValueError(f'`local_world_size` should be >= 1, got {local_world_size}.')
    cpu_count = _num_cpus_available()
    return max(1, cpu_count // local_world_size - 1)