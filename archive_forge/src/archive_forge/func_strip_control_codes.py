import sys
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Union
from .segment import ControlCode, ControlType, Segment
def strip_control_codes(text: str, _translate_table: Dict[int, None]=_CONTROL_STRIP_TRANSLATE) -> str:
    """Remove control codes from text.

    Args:
        text (str): A string possibly contain control codes.

    Returns:
        str: String with control codes removed.
    """
    return text.translate(_translate_table)