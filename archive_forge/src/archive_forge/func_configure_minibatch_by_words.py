import itertools
from functools import partial
from typing import (
from ..util import minibatch, registry
@registry.batchers('spacy.batch_by_words.v1')
def configure_minibatch_by_words(*, size: Sizing, tolerance: float, discard_oversize: bool, get_length: Optional[Callable[[ItemT], int]]=None) -> BatcherT:
    """Create a batcher that uses the "minibatch by words" strategy.

    size (int or Sequence[int]): The target number of words per batch.
        Can be a single integer, or a sequence, allowing for variable batch sizes.
    tolerance (float): What percentage of the size to allow batches to exceed.
    discard_oversize (bool): Whether to discard sequences that by themselves
        exceed the tolerated size.
    get_length (Callable or None): Function to get the length of a sequence
        item. The `len` function is used by default.
    """
    optionals = {'get_length': get_length} if get_length is not None else {}
    return partial(minibatch_by_words, size=size, tolerance=tolerance, discard_oversize=discard_oversize, **optionals)