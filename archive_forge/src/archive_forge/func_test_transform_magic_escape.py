import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_transform_magic_escape():
    check_transform(ipt2.EscapedCommand, MULTILINE_MAGIC)
    check_transform(ipt2.EscapedCommand, INDENTED_MAGIC)
    check_transform(ipt2.EscapedCommand, CRLF_MAGIC)