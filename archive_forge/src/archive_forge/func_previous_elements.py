import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
@property
def previous_elements(self):
    """All PageElements that were parsed before this one.

        :yield: A sequence of PageElements.
        """
    i = self.previous_element
    while i is not None:
        yield i
        i = i.previous_element