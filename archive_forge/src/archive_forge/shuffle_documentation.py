from typing import Any, Dict, List, Optional, Tuple, Union
from ray.data._internal.block_list import BlockList
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import Block, BlockMetadata

        Reduce function to be run for each output block.

        Args:
            mapper_outputs: List of blocks to reduce.
            partial_reduce: A flag passed by the shuffle operator that
                indicates whether we should partially or fully reduce the
                mapper outputs.

        Returns:
            The reduced block and its metadata.
        