from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def valid_label_length(label: Union[bytes, str]) -> bool:
    if len(label) > 63:
        return False
    return True