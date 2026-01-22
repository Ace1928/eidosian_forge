import traceback
from collections import deque, namedtuple
from decimal import Decimal
from itertools import chain
from numbers import Number
from pprint import _recursion
from typing import Any, AnyStr, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple  # noqa
from .text import truncate
def reprstream(stack: deque, seen: Optional[Set]=None, maxlevels: int=3, level: int=0, isinstance: Callable=isinstance) -> Iterator[Any]:
    """Streaming repr, yielding tokens."""
    seen = seen or set()
    append = stack.append
    popleft = stack.popleft
    is_in_seen = seen.__contains__
    discard_from_seen = seen.discard
    add_to_seen = seen.add
    while stack:
        lit_start = lit_end = None
        it = popleft()
        for val in it:
            orig = val
            if isinstance(val, _dirty):
                discard_from_seen(val.objid)
                continue
            elif isinstance(val, _literal):
                level += val.direction
                yield (val, it)
            elif isinstance(val, _key):
                yield (val, it)
            elif isinstance(val, Decimal):
                yield (_repr(val), it)
            elif isinstance(val, safe_t):
                yield (str(val), it)
            elif isinstance(val, chars_t):
                yield (_quoted(val), it)
            elif isinstance(val, range):
                yield (_repr(val), it)
            else:
                if isinstance(val, set_t):
                    if not val:
                        yield (_repr_empty_set(val), it)
                        continue
                    lit_start, lit_end, val = _reprseq(val, LIT_SET_START, LIT_SET_END, set, _chainlist)
                elif isinstance(val, tuple):
                    lit_start, lit_end, val = (LIT_TUPLE_START, LIT_TUPLE_END_SV if len(val) == 1 else LIT_TUPLE_END, _chainlist(val))
                elif isinstance(val, dict):
                    lit_start, lit_end, val = (LIT_DICT_START, LIT_DICT_END, _chaindict(val))
                elif isinstance(val, list):
                    lit_start, lit_end, val = (LIT_LIST_START, LIT_LIST_END, _chainlist(val))
                else:
                    yield (_repr(val), it)
                    continue
                if maxlevels and level >= maxlevels:
                    yield (f'{lit_start.value}...{lit_end.value}', it)
                    continue
                objid = id(orig)
                if is_in_seen(objid):
                    yield (_recursion(orig), it)
                    continue
                add_to_seen(objid)
                append(chain([lit_start], val, [_dirty(objid), lit_end], it))
                break