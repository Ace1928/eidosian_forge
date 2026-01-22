from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
def test_nonlocal(self):
    with self.assertRaises(SyntaxError):
        iprc('nonlocal x')
        iprc('\n            x = 1\n            def f():\n                nonlocal x\n                x = 10000\n                yield x\n            ')
        iprc('\n            def f():\n                def g():\n                    nonlocal x\n                    x = 10000\n                    yield x\n            ')
    iprc('\n        def f():\n            x = 20\n            def g():\n                nonlocal x\n                x = 10000\n                yield x\n        ')