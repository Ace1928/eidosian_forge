from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def check_errno_is_5(e: OSError) -> bool:
    return e.errno == 5