import itertools
from typing import Any, Dict, Iterable, List, Tuple
def safe_iter(it: Iterable[Any], safe: bool=True) -> Iterable[Any]:
    if not safe:
        yield from it
    else:
        n = 0
        for x in it:
            yield x
            n += 1
        if n == 0:
            yield _EMPTY_ITER