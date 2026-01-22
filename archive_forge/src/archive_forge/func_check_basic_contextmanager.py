from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_basic_contextmanager() -> None:
    with RaisesGroup(ValueError) as e:
        raise ExceptionGroup('foo', (ValueError(),))
    assert_type(e.value, BaseExceptionGroup[ValueError])