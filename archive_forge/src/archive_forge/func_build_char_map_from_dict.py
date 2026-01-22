from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def build_char_map_from_dict(space_width: float, ft: DictionaryObject) -> Tuple[str, float, Union[str, Dict[int, str]], Dict[Any, Any]]:
    """
    Determine information about a font.

    Args:
        space_width: default space with if no data found
             (normally half the width of a character).
        ft: Font Dictionary

    Returns:
        Font sub-type, space_width criteria(50% of width), encoding, map character-map.
        The font-dictionary itself is suitable for the curious.
    """
    font_type: str = cast(str, ft['/Subtype'])
    space_code = 32
    encoding, space_code = parse_encoding(ft, space_code)
    map_dict, space_code, int_entry = parse_to_unicode(ft, space_code)
    if encoding == '':
        if -1 not in map_dict or map_dict[-1] == 1:
            encoding = 'charmap'
        else:
            encoding = 'utf-16-be'
    elif isinstance(encoding, dict):
        for x in int_entry:
            if x <= 255:
                encoding[x] = chr(x)
    try:
        space_width = _default_fonts_space_width[cast(str, ft['/BaseFont'])]
    except Exception:
        pass
    if isinstance(space_code, str):
        try:
            sp = space_code.encode('charmap')[0]
        except Exception:
            sp = space_code.encode('utf-16-be')
            sp = sp[0] + 256 * sp[1]
    else:
        sp = space_code
    sp_width = compute_space_width(ft, sp, space_width)
    return (font_type, float(sp_width / 2), encoding, map_dict)