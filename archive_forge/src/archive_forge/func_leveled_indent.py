import collections
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import numpy as np
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.util import capfirst
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def leveled_indent(lvl: int=0, spaces_per_indent: int=3) -> str:
    """Returns a string of spaces which contains `level` indents,
    each indent containing `spaces_per_indent` spaces. For example:
    >>> leveled_indent(2, 3)
    '      '
    """
    return ' ' * spaces_per_indent * lvl