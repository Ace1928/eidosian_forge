import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def restore_byte_a0(byts):
    """
    Some mojibake has been additionally altered by a process that said "hmm,
    byte A0, that's basically a space!" and replaced it with an ASCII space.
    When the A0 is part of a sequence that we intend to decode as UTF-8,
    changing byte A0 to 20 would make it fail to decode.

    This process finds sequences that would convincingly decode as UTF-8 if
    byte 20 were changed to A0, and puts back the A0. For the purpose of
    deciding whether this is a good idea, this step gets a cost of twice
    the number of bytes that are changed.

    This is used as a step within `fix_encoding`.
    """
    byts = A_GRAVE_WORD_RE.sub(b'\xc3\xa0 ', byts)

    def replacement(match):
        """The function to apply when this regex matches."""
        return match.group(0).replace(b' ', b'\xa0')
    return ALTERED_UTF8_RE.sub(replacement, byts)