import textwrap
from unicodedata import east_asian_width as _eawidth
from . import osutils
_fix_sentence_endings(chunks : [string])

        Correct for sentence endings buried in 'chunks'.  Eg. when the
        original text contains "... foo.
Bar ...", munge_whitespace()
        and split() will convert that to [..., "foo.", " ", "Bar", ...]
        which has one too few spaces; this method simply changes the one
        space to two.

        Note: This function is copied from textwrap.TextWrap and modified
        to use unicode always.
        