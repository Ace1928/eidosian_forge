from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def simple_table(rows: list[tuple[str, str]], headers: tuple[str, str], **kwargs):
    """
    Generate a simple markdown table

    The header is center aligned
    The cells is left aligned
    """
    column_width = [len(s) + 2 for s in headers]
    for row in rows:
        for i, cell in enumerate(row):
            column_width[i] = max(column_width[i], len(cell))
    sep = '  '
    underline = sep.join(('-' * w for w in column_width))
    formatting_spec = sep.join((f'{{{i}: <{w}}}' for i, w in enumerate(column_width)))
    format_row = formatting_spec.format
    format_header = formatting_spec.replace('<', '^').format
    _rows = [format_header(*headers), underline, *[format_row(*row) for row in rows]]
    return '\n'.join(_rows)