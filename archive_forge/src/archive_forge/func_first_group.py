import sys
from Cython.Tempita.compat3 import basestring_
def first_group(self, getter=None):
    """
        Returns true if this item is the start of a new group,
        where groups mean that some attribute has changed.  The getter
        can be None (the item itself changes), an attribute name like
        ``'.attr'``, a function, or a dict key or list index.
        """
    if self.first:
        return True
    return self._compare_group(self.item, self.previous, getter)