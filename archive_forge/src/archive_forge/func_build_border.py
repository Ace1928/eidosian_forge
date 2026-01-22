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
def build_border(self, props: Mapping[str, str]) -> dict[str, dict[str, str | None]]:
    return {side: {'style': self._border_style(props.get(f'border-{side}-style'), props.get(f'border-{side}-width'), self.color_to_excel(props.get(f'border-{side}-color'))), 'color': self.color_to_excel(props.get(f'border-{side}-color'))} for side in ['top', 'right', 'bottom', 'left']}