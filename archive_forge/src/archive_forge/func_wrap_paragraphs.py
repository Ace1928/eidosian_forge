from __future__ import annotations
import re
import textwrap
from textwrap import dedent
from textwrap import indent as _indent
from typing import List
def wrap_paragraphs(text: str, ncols: int=80) -> List[str]:
    """Wrap multiple paragraphs to fit a specified width.

    This is equivalent to textwrap.wrap, but with support for multiple
    paragraphs, as separated by empty lines.

    Returns
    -------

    list of complete paragraphs, wrapped to fill `ncols` columns.
    """
    paragraph_re = re.compile('\\n(\\s*\\n)+', re.MULTILINE)
    text = dedent(text).strip()
    paragraphs = paragraph_re.split(text)[::2]
    out_ps = []
    indent_re = re.compile('\\n\\s+', re.MULTILINE)
    for p in paragraphs:
        if indent_re.search(p) is None:
            p = textwrap.fill(p, ncols)
        out_ps.append(p)
    return out_ps