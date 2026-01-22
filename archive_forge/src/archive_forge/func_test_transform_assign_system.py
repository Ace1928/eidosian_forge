import platform
import string
import sys
from textwrap import dedent
import pytest
from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line
def test_transform_assign_system():
    check_transform(ipt2.SystemAssign, MULTILINE_SYSTEM_ASSIGN)
    check_transform(ipt2.SystemAssign, MULTILINE_SYSTEM_ASSIGN_AFTER_DEDENT)