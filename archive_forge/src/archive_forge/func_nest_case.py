from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
def nest_case(context, case):
    lines = context.strip().splitlines()
    prefix_len = 0
    for c in lines[-1]:
        if c != ' ':
            break
        prefix_len += 1
    indented_case = indent(case, ' ' * (prefix_len + 4))
    return context + '\n' + indented_case