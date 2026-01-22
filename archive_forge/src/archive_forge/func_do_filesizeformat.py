import math
import random
import re
import typing
import typing as t
from collections import abc
from itertools import chain
from itertools import groupby
from markupsafe import escape
from markupsafe import Markup
from markupsafe import soft_str
from .async_utils import async_variant
from .async_utils import auto_aiter
from .async_utils import auto_await
from .async_utils import auto_to_list
from .exceptions import FilterArgumentError
from .runtime import Undefined
from .utils import htmlsafe_json_dumps
from .utils import pass_context
from .utils import pass_environment
from .utils import pass_eval_context
from .utils import pformat
from .utils import url_quote
from .utils import urlize
def do_filesizeformat(value: t.Union[str, float, int], binary: bool=False) -> str:
    """Format the value like a 'human-readable' file size (i.e. 13 kB,
    4.1 MB, 102 Bytes, etc).  Per default decimal prefixes are used (Mega,
    Giga, etc.), if the second parameter is set to `True` the binary
    prefixes are used (Mebi, Gibi).
    """
    bytes = float(value)
    base = 1024 if binary else 1000
    prefixes = ['KiB' if binary else 'kB', 'MiB' if binary else 'MB', 'GiB' if binary else 'GB', 'TiB' if binary else 'TB', 'PiB' if binary else 'PB', 'EiB' if binary else 'EB', 'ZiB' if binary else 'ZB', 'YiB' if binary else 'YB']
    if bytes == 1:
        return '1 Byte'
    elif bytes < base:
        return f'{int(bytes)} Bytes'
    else:
        for i, prefix in enumerate(prefixes):
            unit = base ** (i + 2)
            if bytes < unit:
                return f'{base * bytes / unit:.1f} {prefix}'
        return f'{base * bytes / unit:.1f} {prefix}'