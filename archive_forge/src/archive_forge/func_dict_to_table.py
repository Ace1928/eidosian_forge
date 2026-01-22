from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
def dict_to_table(header: tuple[str, str], contents: dict[str, str]) -> str:
    """
    Convert dict to an (n x 2) table

    Parameters
    ----------
    header : tuple
        Table header. Should have a length of 2.
    contents : dict
        The key becomes column 1 of table and the
        value becomes column 2 of table.

    Examples
    --------
    >>> d = {"alpha": 1, "color": "blue", "fill": None}
    >>> print(dict_to_table(("Aesthetic", "Default Value"), d))
    Aesthetic  Default Value
    ---------  -------------
    alpha      `1`
    color      `'blue'`
    fill       `None`
    """
    rows = [(name, value if value == '' else f'`{value!r}`{{.py}}') for name, value in contents.items()]
    return table_function(rows, headers=header, tablefmt='grid')