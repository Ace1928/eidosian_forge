from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def write_td(self, s: Any, indent: int=0, tags: str | None=None) -> None:
    self._write_cell(s, kind='td', indent=indent, tags=tags)