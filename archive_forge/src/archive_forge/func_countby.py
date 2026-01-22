import itertools
from .itertoolz import frequencies, pluck, getter
def countby(key, seq):
    """ Count elements of a collection by a key function

    >>> countby(len, ['cat', 'mouse', 'dog'])
    {3: 2, 5: 1}

    >>> def iseven(x): return x % 2 == 0
    >>> countby(iseven, [1, 2, 3])  # doctest:+SKIP
    {True: 1, False: 2}

    See Also:
        groupby
    """
    if not callable(key):
        key = getter(key)
    return frequencies(map(key, seq))