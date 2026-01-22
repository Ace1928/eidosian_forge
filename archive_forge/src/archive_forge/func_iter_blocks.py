import math
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats, _get_or_create_stats_actor
from ray.data._internal.util import _split_list
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def iter_blocks(self) -> Iterator[ObjectRef[Block]]:
    """Iterate over the blocks of this block list.

        This blocks on the execution of the tasks generating block outputs.
        The length of this iterator is not known until execution.
        """
    self._check_if_cleared()
    outer = self

    class Iter:

        def __init__(self):
            self._base_iter = outer.iter_blocks_with_metadata()

        def __iter__(self):
            return self

        def __next__(self):
            ref, meta = next(self._base_iter)
            assert isinstance(ref, ray.ObjectRef), (ref, meta)
            return ref
    return Iter()