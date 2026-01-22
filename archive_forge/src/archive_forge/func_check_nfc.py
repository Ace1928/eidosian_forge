from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def check_nfc(label: str) -> None:
    if unicodedata.normalize('NFC', label) != label:
        raise IDNAError('Label must be in Normalization Form C')