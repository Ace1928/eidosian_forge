import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from shutil import which
from typing import List, Optional
import torch
from packaging.version import parse
@lru_cache
def set_numa_affinity(local_process_index: int, verbose: Optional[bool]=None) -> None:
    """
    Assigns the current process to a specific NUMA node. Ideally most efficient when having at least 2 cpus per node.

    This result is cached between calls. If you want to override it, please use
    `accelerate.utils.environment.override_numa_afifnity`.

    Args:
        local_process_index (int):
            The index of the current process on the current server.
        verbose (bool, *optional*):
            Whether to print the new cpu cores assignment for each process. If `ACCELERATE_DEBUG_MODE` is enabled, will
            default to True.
    """
    override_numa_affinity(local_process_index=local_process_index, verbose=verbose)