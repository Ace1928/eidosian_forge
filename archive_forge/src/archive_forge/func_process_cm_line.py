from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def process_cm_line(line: bytes, process_rg: bool, process_char: bool, multiline_rg: Union[None, Tuple[int, int]], map_dict: Dict[Any, Any], int_entry: List[int]) -> Tuple[bool, bool, Union[None, Tuple[int, int]]]:
    if line == b'' or line[0] == 37:
        return (process_rg, process_char, multiline_rg)
    line = line.replace(b'\t', b' ')
    if b'beginbfrange' in line:
        process_rg = True
    elif b'endbfrange' in line:
        process_rg = False
    elif b'beginbfchar' in line:
        process_char = True
    elif b'endbfchar' in line:
        process_char = False
    elif process_rg:
        multiline_rg = parse_bfrange(line, map_dict, int_entry, multiline_rg)
    elif process_char:
        parse_bfchar(line, map_dict, int_entry)
    return (process_rg, process_char, multiline_rg)