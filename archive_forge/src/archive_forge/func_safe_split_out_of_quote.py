from typing import Iterable, List, Tuple
from triad.constants import TRIAD_VAR_QUOTE
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def safe_split_out_of_quote(s: str, sep_chars: str, max_split: int=-1, quote: str=TRIAD_VAR_QUOTE) -> List[str]:
    b = 0
    if max_split == 0 or len(s) == 0:
        return [s]
    res: List[str] = []
    for p, _ in safe_search_out_of_quote(s, sep_chars, quote=quote):
        res.append(s[b:p])
        b = p + 1
        if len(res) == max_split:
            break
    res.append(s[b:])
    return res