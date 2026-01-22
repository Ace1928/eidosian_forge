from __future__ import annotations
import re
import importlib.util
import sys
from typing import TYPE_CHECKING, Sequence
def get_endtag_text(self, tag: str) -> str:
    """
        Returns the text of the end tag.

        If it fails to extract the actual text from the raw data, it builds a closing tag with `tag`.
        """
    start = self.line_offset + self.offset
    m = htmlparser.endendtag.search(self.rawdata, start)
    if m:
        return self.rawdata[start:m.end()]
    else:
        return '</{}>'.format(tag)