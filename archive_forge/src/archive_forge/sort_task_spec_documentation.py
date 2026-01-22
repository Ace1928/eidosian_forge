from typing import List, Tuple, TypeVar, Union
import numpy as np
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.sort import SortKey
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.types import ObjectRef

        Return (num_reducers - 1) items in ascending order from the blocks that
        partition the domain into ranges with approximately equally many elements.
        Each boundary item is a tuple of a form (col1_value, col2_value, ...).
        