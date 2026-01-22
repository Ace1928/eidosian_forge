from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def check_message(message: str, body: RaisesGroup[Any]) -> None:
    with pytest.raises(AssertionError, match=f'^DID NOT RAISE any exception, expected {re.escape(message)}$'):
        with body:
            ...