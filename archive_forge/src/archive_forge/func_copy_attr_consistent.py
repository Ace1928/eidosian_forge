import sys
import os
import re
import warnings
import types
import unicodedata
def copy_attr_consistent(self, attr, value, replace):
    """
        If replace is True or self[attr] is None, replace self[attr] with
        value.  Otherwise, do nothing.
        """
    if self.get(attr) is not value:
        self.replace_attr(attr, value, replace)