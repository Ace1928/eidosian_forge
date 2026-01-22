from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
def object_was_parsed(self, o, parent=None, most_recent_element=None):
    """Method called by the TreeBuilder to integrate an object into the parse tree."""
    if parent is None:
        parent = self.currentTag
    if most_recent_element is not None:
        previous_element = most_recent_element
    else:
        previous_element = self._most_recent_element
    next_element = previous_sibling = next_sibling = None
    if isinstance(o, Tag):
        next_element = o.next_element
        next_sibling = o.next_sibling
        previous_sibling = o.previous_sibling
        if previous_element is None:
            previous_element = o.previous_element
    fix = parent.next_element is not None
    o.setup(parent, previous_element, next_element, previous_sibling, next_sibling)
    self._most_recent_element = o
    parent.contents.append(o)
    if fix:
        self._linkage_fixer(parent)