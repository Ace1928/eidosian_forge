from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def build_char_map(font_name: str, space_width: float, obj: DictionaryObject) -> Tuple[str, float, Union[str, Dict[int, str]], Dict[Any, Any], DictionaryObject]:
    """
    Determine information about a font.

    Args:
        font_name: font name as a string
        space_width: default space width if no data is found.
        obj: XObject or Page where you can find a /Resource dictionary

    Returns:
        Font sub-type, space_width criteria (50% of width), encoding, map character-map, font-dictionary.
        The font-dictionary itself is suitable for the curious.
    """
    ft: DictionaryObject = obj['/Resources']['/Font'][font_name]
    font_subtype, font_halfspace, font_encoding, font_map = build_char_map_from_dict(space_width, ft)
    return (font_subtype, font_halfspace, font_encoding, font_map, ft)