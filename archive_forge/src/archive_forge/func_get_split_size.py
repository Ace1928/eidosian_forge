from typing import List, Optional, Tuple
from torch.distributed._shard.metadata import ShardMetadata
def get_split_size(dim_size, chunks):
    """
    Computes the split size inline with ``torch.chunk``

    Args:
        dim_size(int): Size of the dimension being chunked.
        chunks(int): Number of chunks to create for ``dim_size``.

    Returns:
        An int indicating the split size to use.
    """
    return (dim_size + chunks - 1) // chunks