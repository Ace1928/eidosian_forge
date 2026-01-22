from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def test_matcher_match() -> None:
    with RaisesGroup(Matcher(ValueError, 'foo')):
        raise ExceptionGroup('', (ValueError('foo'),))
    with pytest.raises(ExceptionGroup):
        with RaisesGroup(Matcher(ValueError, 'foo')):
            raise ExceptionGroup('', (ValueError('bar'),))
    with RaisesGroup(Matcher(match='foo')):
        raise ExceptionGroup('', (ValueError('foo'),))
    with pytest.raises(ExceptionGroup):
        with RaisesGroup(Matcher(match='foo')):
            raise ExceptionGroup('', (ValueError('bar'),))