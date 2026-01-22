from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
def doesNotRaiseExpected(*args, **kwargs):
    raise _UnexpectedException