import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ..generic import DictionaryObject, TextStringObject, encode_pdfdocencoding
def set_custom_rtl(_min: Union[str, int, None]=None, _max: Union[str, int, None]=None, specials: Union[str, List[int], None]=None) -> Tuple[int, int, List[int]]:
    """
    Change the Right-To-Left and special characters custom parameters.

    Args:
        _min: The new minimum value for the range of custom characters that
            will be written right to left.
            If set to ``None``, the value will not be changed.
            If set to an integer or string, it will be converted to its ASCII code.
            The default value is -1, which sets no additional range to be converted.
        _max: The new maximum value for the range of custom characters that will
            be written right to left.
            If set to ``None``, the value will not be changed.
            If set to an integer or string, it will be converted to its ASCII code.
            The default value is -1, which sets no additional range to be converted.
        specials: The new list of special characters to be inserted in the
            current insertion order.
            If set to ``None``, the current value will not be changed.
            If set to a string, it will be converted to a list of ASCII codes.
            The default value is an empty list.

    Returns:
        A tuple containing the new values for ``CUSTOM_RTL_MIN``,
        ``CUSTOM_RTL_MAX``, and ``CUSTOM_RTL_SPECIAL_CHARS``.
    """
    global CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS
    if isinstance(_min, int):
        CUSTOM_RTL_MIN = _min
    elif isinstance(_min, str):
        CUSTOM_RTL_MIN = ord(_min)
    if isinstance(_max, int):
        CUSTOM_RTL_MAX = _max
    elif isinstance(_max, str):
        CUSTOM_RTL_MAX = ord(_max)
    if isinstance(specials, str):
        CUSTOM_RTL_SPECIAL_CHARS = [ord(x) for x in specials]
    elif isinstance(specials, list):
        CUSTOM_RTL_SPECIAL_CHARS = specials
    return (CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS)