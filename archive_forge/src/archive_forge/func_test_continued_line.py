import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_continued_line():
    lines = MULTILINE_MAGIC_ASSIGN[0]
    assert ipt2.find_end_of_continued_line(lines, 1) == 2
    assert ipt2.assemble_continued_line(lines, (1, 5), 2) == 'foo    bar'