import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def restore_default(self, key):
    """
        Restore (and return) default value for the specified key.
        
        This method will only work for a ConfigObj that was created
        with a configspec and has been validated.
        
        If there is no default value for this key, ``KeyError`` is raised.
        """
    default = self.default_values[key]
    dict.__setitem__(self, key, default)
    if key not in self.defaults:
        self.defaults.append(key)
    return default