import os
import stat
import sys
import typing as t
from datetime import datetime
from gettext import gettext as _
from gettext import ngettext
from ._compat import _get_argv_encoding
from ._compat import open_stream
from .exceptions import BadParameter
from .utils import format_filename
from .utils import LazyFile
from .utils import safecall
def coerce_path_result(self, value: 't.Union[str, os.PathLike[str]]') -> 't.Union[str, bytes, os.PathLike[str]]':
    if self.type is not None and (not isinstance(value, self.type)):
        if self.type is str:
            return os.fsdecode(value)
        elif self.type is bytes:
            return os.fsencode(value)
        else:
            return t.cast('os.PathLike[str]', self.type(value))
    return value