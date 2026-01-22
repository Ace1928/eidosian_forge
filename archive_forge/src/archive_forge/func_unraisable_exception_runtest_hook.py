import sys
import traceback
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Generator
from typing import Optional
from typing import Type
import warnings
import pytest
def unraisable_exception_runtest_hook() -> Generator[None, None, None]:
    with catch_unraisable_exception() as cm:
        try:
            yield
        finally:
            if cm.unraisable:
                if cm.unraisable.err_msg is not None:
                    err_msg = cm.unraisable.err_msg
                else:
                    err_msg = 'Exception ignored in'
                msg = f'{err_msg}: {cm.unraisable.object!r}\n\n'
                msg += ''.join(traceback.format_exception(cm.unraisable.exc_type, cm.unraisable.exc_value, cm.unraisable.exc_traceback))
                warnings.warn(pytest.PytestUnraisableExceptionWarning(msg))