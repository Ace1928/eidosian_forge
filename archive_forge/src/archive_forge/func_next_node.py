import sys
import os
import re
import warnings
import types
import unicodedata
def next_node(self, condition=None, include_self=False, descend=True, siblings=False, ascend=False):
    """
        Return the first node in the iterable returned by traverse(),
        or None if the iterable is empty.

        Parameter list is the same as of traverse.  Note that
        include_self defaults to 0, though.
        """
    iterable = self.traverse(condition=condition, include_self=include_self, descend=descend, siblings=siblings, ascend=ascend)
    try:
        return iterable[0]
    except IndexError:
        return None