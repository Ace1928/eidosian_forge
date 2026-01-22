from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def parse_to_unicode(ft: DictionaryObject, space_code: int) -> Tuple[Dict[Any, Any], int, List[int]]:
    map_dict: Dict[Any, Any] = {}
    int_entry: List[int] = []
    if '/ToUnicode' not in ft:
        if ft.get('/Subtype', '') == '/Type1':
            return type1_alternative(ft, map_dict, space_code, int_entry)
        else:
            return ({}, space_code, [])
    process_rg: bool = False
    process_char: bool = False
    multiline_rg: Union[None, Tuple[int, int]] = None
    cm = prepare_cm(ft)
    for line in cm.split(b'\n'):
        process_rg, process_char, multiline_rg = process_cm_line(line.strip(b' \t'), process_rg, process_char, multiline_rg, map_dict, int_entry)
    for a, value in map_dict.items():
        if value == ' ':
            space_code = a
    return (map_dict, space_code, int_entry)