import re
from functools import partial, reduce
from math import gcd
from operator import itemgetter
from typing import (
from ._loop import loop_last
from ._pick import pick_bool
from ._wrap import divide_line
from .align import AlignMethod
from .cells import cell_len, set_cell_size
from .containers import Lines
from .control import strip_control_codes
from .emoji import EmojiVariant
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import Style, StyleType
def highlight_words(self, words: Iterable[str], style: Union[str, Style], *, case_sensitive: bool=True) -> int:
    """Highlight words with a style.

        Args:
            words (Iterable[str]): Worlds to highlight.
            style (Union[str, Style]): Style to apply.
            case_sensitive (bool, optional): Enable case sensitive matchings. Defaults to True.

        Returns:
            int: Number of words highlighted.
        """
    re_words = '|'.join((re.escape(word) for word in words))
    add_span = self._spans.append
    count = 0
    _Span = Span
    for match in re.finditer(re_words, self.plain, flags=0 if case_sensitive else re.IGNORECASE):
        start, end = match.span(0)
        add_span(_Span(start, end, style))
        count += 1
    return count