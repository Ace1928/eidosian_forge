import sys
import os
import re
import pkg_resources
from string import Template
def normalize_pkg_name(self, dest):
    return _bad_chars_re.sub('', dest.lower())