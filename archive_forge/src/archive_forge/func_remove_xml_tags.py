import logging
import re
from typing import Optional, Union
from .enums import LanguageFilter, ProbingState
@staticmethod
def remove_xml_tags(buf: Union[bytes, bytearray]) -> bytes:
    """
        Returns a copy of ``buf`` that retains only the sequences of English
        alphabet and high byte characters that are not between <> characters.
        This filter can be applied to all scripts which contain both English
        characters and extended ASCII characters, but is currently only used by
        ``Latin1Prober``.
        """
    filtered = bytearray()
    in_tag = False
    prev = 0
    buf = memoryview(buf).cast('c')
    for curr, buf_char in enumerate(buf):
        if buf_char == b'>':
            prev = curr + 1
            in_tag = False
        elif buf_char == b'<':
            if curr > prev and (not in_tag):
                filtered.extend(buf[prev:curr])
                filtered.extend(b' ')
            in_tag = True
    if not in_tag:
        filtered.extend(buf[prev:])
    return filtered