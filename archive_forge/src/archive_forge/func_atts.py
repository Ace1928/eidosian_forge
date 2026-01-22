import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
@property
def atts(self) -> FrozenAttributes:
    """Attributes, e.g. {'fg': 34, 'bold': True} where 34 is the escape code for ..."""
    return self._atts