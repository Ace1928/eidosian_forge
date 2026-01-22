from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def parse_bfchar(line: bytes, map_dict: Dict[Any, Any], int_entry: List[int]) -> None:
    lst = [x for x in line.split(b' ') if x]
    map_dict[-1] = len(lst[0]) // 2
    while len(lst) > 1:
        map_to = ''
        if lst[1] != b'.':
            map_to = unhexlify(lst[1]).decode('charmap' if len(lst[1]) < 4 else 'utf-16-be', 'surrogatepass')
        map_dict[unhexlify(lst[0]).decode('charmap' if map_dict[-1] == 1 else 'utf-16-be', 'surrogatepass')] = map_to
        int_entry.append(int(lst[0], 16))
        lst = lst[2:]