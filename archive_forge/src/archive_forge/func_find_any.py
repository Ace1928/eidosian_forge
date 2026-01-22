import collections
import copy
import itertools
import random
import re
import warnings
def find_any(self, *args, **kwargs):
    """Return the first element found by find_elements(), or None.

        This is also useful for checking whether any matching element exists in
        the tree, and can be used in a conditional expression.
        """
    hits = self.find_elements(*args, **kwargs)
    try:
        return next(hits)
    except StopIteration:
        return None