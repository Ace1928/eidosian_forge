import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
@property
def previous_siblings(self):
    """All PageElements that are siblings of this one but were parsed
        earlier.

        :yield: A sequence of PageElements.
        """
    i = self.previous_sibling
    while i is not None:
        yield i
        i = i.previous_sibling