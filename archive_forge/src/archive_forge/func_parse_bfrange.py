from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def parse_bfrange(line: bytes, map_dict: Dict[Any, Any], int_entry: List[int], multiline_rg: Union[None, Tuple[int, int]]) -> Union[None, Tuple[int, int]]:
    lst = [x for x in line.split(b' ') if x]
    closure_found = False
    if multiline_rg is not None:
        fmt = b'%%0%dX' % (map_dict[-1] * 2)
        a = multiline_rg[0]
        b = multiline_rg[1]
        for sq in lst[0:]:
            if sq == b']':
                closure_found = True
                break
            map_dict[unhexlify(fmt % a).decode('charmap' if map_dict[-1] == 1 else 'utf-16-be', 'surrogatepass')] = unhexlify(sq).decode('utf-16-be', 'surrogatepass')
            int_entry.append(a)
            a += 1
    else:
        a = int(lst[0], 16)
        b = int(lst[1], 16)
        nbi = max(len(lst[0]), len(lst[1]))
        map_dict[-1] = ceil(nbi / 2)
        fmt = b'%%0%dX' % (map_dict[-1] * 2)
        if lst[2] == b'[':
            for sq in lst[3:]:
                if sq == b']':
                    closure_found = True
                    break
                map_dict[unhexlify(fmt % a).decode('charmap' if map_dict[-1] == 1 else 'utf-16-be', 'surrogatepass')] = unhexlify(sq).decode('utf-16-be', 'surrogatepass')
                int_entry.append(a)
                a += 1
        else:
            c = int(lst[2], 16)
            fmt2 = b'%%0%dX' % max(4, len(lst[2]))
            closure_found = True
            while a <= b:
                map_dict[unhexlify(fmt % a).decode('charmap' if map_dict[-1] == 1 else 'utf-16-be', 'surrogatepass')] = unhexlify(fmt2 % c).decode('utf-16-be', 'surrogatepass')
                int_entry.append(a)
                a += 1
                c += 1
    return None if closure_found else (a, b)