import sys
import os
import re
import warnings
import types
import unicodedata
def copy_attr_concatenate(self, attr, value, replace):
    """
        If attr is an attribute of self and both self[attr] and value are
        lists, concatenate the two sequences, setting the result to
        self[attr].  If either self[attr] or value are non-sequences and
        replace is True or self[attr] is None, replace self[attr] with value.
        Otherwise, do nothing.
        """
    if self.get(attr) is not value:
        if isinstance(self.get(attr), list) and isinstance(value, list):
            self.append_attr_list(attr, value)
        else:
            self.replace_attr(attr, value, replace)