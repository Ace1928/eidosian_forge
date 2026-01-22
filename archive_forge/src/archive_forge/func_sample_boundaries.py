from typing import List, Tuple, TypeVar, Union
import numpy as np
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.sort import SortKey
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.types import ObjectRef
@staticmethod
def sample_boundaries(blocks: List[ObjectRef[Block]], sort_key: SortKey, num_reducers: int) -> List[T]:
    """
        Return (num_reducers - 1) items in ascending order from the blocks that
        partition the domain into ranges with approximately equally many elements.
        Each boundary item is a tuple of a form (col1_value, col2_value, ...).
        """
    columns = sort_key.get_columns()
    n_samples = int(num_reducers * 10 / len(blocks))
    sample_block = cached_remote_fn(_sample_block)
    sample_results = [sample_block.remote(block, n_samples, sort_key) for block in blocks]
    sample_bar = ProgressBar(SortTaskSpec.SORT_SAMPLE_SUB_PROGRESS_BAR_NAME, len(sample_results))
    samples = sample_bar.fetch_until_complete(sample_results)
    sample_bar.close()
    del sample_results
    samples = [s for s in samples if len(s) > 0]
    if len(samples) == 0:
        return [None] * (num_reducers - 1)
    builder = DelegatingBlockBuilder()
    for sample in samples:
        builder.add_block(sample)
    samples = builder.build()
    sample_dict = BlockAccessor.for_block(samples).to_numpy(columns=columns)
    indices = np.lexsort(list(reversed(list(sample_dict.values()))))
    for k, v in sample_dict.items():
        sorted_v = v[indices]
        sample_dict[k] = list(np.quantile(sorted_v, np.linspace(0, 1, num_reducers), interpolation='nearest')[1:])
    return [tuple((sample_dict[k][i] for k in sample_dict)) for i in range(num_reducers - 1)]