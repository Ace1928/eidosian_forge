import math
from itertools import islice
from nltk.util import choose, ngrams
def find_increasing_sequences(worder):
    """
    Given the *worder* list, this function groups monotonic +1 sequences.

        >>> worder = [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]
        >>> list(find_increasing_sequences(worder))
        [(7, 8, 9, 10), (0, 1, 2, 3, 4, 5)]

    :param worder: The worder list output from word_rank_alignment
    :param type: list(int)
    """
    items = iter(worder)
    a, b = (None, next(items, None))
    result = [b]
    while b is not None:
        a, b = (b, next(items, None))
        if b is not None and a + 1 == b:
            result.append(b)
        else:
            if len(result) > 1:
                yield tuple(result)
            result = [b]