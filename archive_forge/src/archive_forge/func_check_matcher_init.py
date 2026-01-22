from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_matcher_init() -> None:

    def check_exc(exc: BaseException) -> bool:
        return isinstance(exc, ValueError)

    def check_filenotfound(exc: FileNotFoundError) -> bool:
        return not exc.filename.endswith('.tmp')
    Matcher()
    Matcher(ValueError)
    Matcher(ValueError, 'regex')
    Matcher(ValueError, 'regex', check_exc)
    Matcher(exception_type=ValueError)
    Matcher(match='regex')
    Matcher(check=check_exc)
    Matcher(check=check_filenotfound)
    Matcher(ValueError, match='regex')
    Matcher(FileNotFoundError, check=check_filenotfound)
    Matcher(match='regex', check=check_exc)
    Matcher(FileNotFoundError, match='regex', check=check_filenotfound)