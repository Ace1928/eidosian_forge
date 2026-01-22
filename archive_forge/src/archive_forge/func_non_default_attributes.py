import sys
import os
import re
import warnings
import types
import unicodedata
def non_default_attributes(self):
    atts = {}
    for key, value in list(self.attributes.items()):
        if self.is_not_default(key):
            atts[key] = value
    return atts