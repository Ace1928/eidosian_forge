import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
@cached_property
def color_str(self) -> str:
    """Return an escape-coded string to write to the terminal."""
    s = self._s
    for k, v in sorted(self._atts.items()):
        if k not in one_arg_xforms and k not in two_arg_xforms:
            continue
        elif v is False:
            continue
        elif k in one_arg_xforms:
            s = one_arg_xforms[k](s)
        else:
            s = two_arg_xforms[k](s, v)
    return s