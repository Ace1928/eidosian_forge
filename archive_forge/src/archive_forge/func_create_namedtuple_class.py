import functools
from collections import namedtuple
@functools.lru_cache
def create_namedtuple_class(*names):

    def __reduce__(self):
        return (unpickle_named_row, (names, tuple(self)))
    return type('Row', (namedtuple('Row', names),), {'__reduce__': __reduce__, '__slots__': ()})