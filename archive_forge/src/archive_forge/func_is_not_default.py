import sys
import os
import re
import warnings
import types
import unicodedata
def is_not_default(self, key):
    if self[key] == [] and key in self.list_attributes:
        return 0
    else:
        return 1