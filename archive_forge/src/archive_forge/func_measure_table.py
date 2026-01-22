import typing as t
from contextlib import contextmanager
from gettext import gettext as _
from ._compat import term_len
from .parser import split_opt
def measure_table(rows: t.Iterable[t.Tuple[str, str]]) -> t.Tuple[int, ...]:
    widths: t.Dict[int, int] = {}
    for row in rows:
        for idx, col in enumerate(row):
            widths[idx] = max(widths.get(idx, 0), term_len(col))
    return tuple((y for x, y in sorted(widths.items())))