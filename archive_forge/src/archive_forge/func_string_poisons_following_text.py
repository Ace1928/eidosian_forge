from __future__ import with_statement
import textwrap
from difflib import ndiff
from io import open
from os import listdir
from os.path import dirname, isdir, join, realpath, relpath, splitext
import pytest
import chardet
@given(st.text(), random=rnd)
@settings(verbosity=Verbosity.quiet, max_shrinks=0, max_examples=50)
def string_poisons_following_text(suffix):
    try:
        extended = (txt + suffix).encode(enc)
    except UnicodeEncodeError:
        assume(False)
    result = chardet.detect(extended)
    if result and result['encoding'] is not None:
        raise JustALengthIssue()