from __future__ import annotations
from collections.abc import (
import functools
import itertools
import re
from typing import (
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes import missing
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import (
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.printing import pprint_thing
def build_font(self, props: Mapping[str, str]) -> dict[str, bool | float | str | None]:
    font_names = self._get_font_names(props)
    decoration = self._get_decoration(props)
    return {'name': font_names[0] if font_names else None, 'family': self._select_font_family(font_names), 'size': self._get_font_size(props), 'bold': self._get_is_bold(props), 'italic': self._get_is_italic(props), 'underline': 'single' if 'underline' in decoration else None, 'strike': 'line-through' in decoration or None, 'color': self.color_to_excel(props.get('color')), 'shadow': self._get_shadow(props)}