import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def partition_to_color(partitions):
    """
    Creates a dictionary that maps each item in each partition to the index of
    the partition to which it belongs.

    Parameters
    ----------
    partitions: collections.abc.Sequence[collections.abc.Iterable]
        As returned by :func:`make_partitions`.

    Returns
    -------
    dict
    """
    colors = {}
    for color, keys in enumerate(partitions):
        for key in keys:
            colors[key] = color
    return colors