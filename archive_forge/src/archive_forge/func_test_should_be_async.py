from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
def test_should_be_async(self):
    self.assertFalse(_should_be_async('False'))
    self.assertTrue(_should_be_async('await bar()'))
    self.assertTrue(_should_be_async('x = await bar()'))
    self.assertFalse(_should_be_async(dedent('\n            async def awaitable():\n                pass\n        ')))