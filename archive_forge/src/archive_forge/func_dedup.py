from typing import Iterable, Any, List
def dedup(seq: List[Any]) -> Iterable[Any]:
    last: Any = None
    for x in sorted(seq):
        if last is None:
            last = x
            yield x
        elif abs(x - last) < error:
            continue
        else:
            last = x
            yield x