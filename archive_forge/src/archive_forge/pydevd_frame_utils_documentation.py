from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict

        The columns internally are actually based on bytes.

        Also, the position isn't always the ideal one as the start may not be
        what we want (if the user has many subscripts in the line the start
        will always be the same and only the end would change).
        For more details see:
        https://github.com/microsoft/debugpy/issues/1099#issuecomment-1303403995

        So, this function maps the start/end columns to the position to be shown in the editor.
        