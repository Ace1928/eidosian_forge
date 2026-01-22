import itertools
import heapq
import collections
import operator
from functools import partial
from itertools import filterfalse, zip_longest
from collections.abc import Sequence
from toolz.utils import no_default
def merge_sorted(*seqs, **kwargs):
    """ Merge and sort a collection of sorted collections

    This works lazily and only keeps one value from each iterable in memory.

    >>> list(merge_sorted([1, 3, 5], [2, 4, 6]))
    [1, 2, 3, 4, 5, 6]

    >>> ''.join(merge_sorted('abc', 'abc', 'abc'))
    'aaabbbccc'

    The "key" function used to sort the input may be passed as a keyword.

    >>> list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    [2, 1, 3, 3]
    """
    if len(seqs) == 0:
        return iter([])
    elif len(seqs) == 1:
        return iter(seqs[0])
    key = kwargs.get('key', None)
    if key is None:
        return _merge_sorted_binary(seqs)
    else:
        return _merge_sorted_binary_key(seqs, key)