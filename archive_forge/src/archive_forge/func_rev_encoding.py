from typing import Dict, List
from .adobe_glyphs import adobe_glyphs
from .pdfdoc import _pdfdoc_encoding
from .std import _std_encoding
from .symbol import _symbol_encoding
from .zapfding import _zapfding_encoding
def rev_encoding(enc: List[str]) -> Dict[str, int]:
    rev: Dict[str, int] = {}
    for i in range(256):
        char = enc[i]
        if char == '\x00':
            continue
        assert char not in rev, f'{char} at {i} already at {rev[char]}'
        rev[char] = i
    return rev