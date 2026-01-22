from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def get_checks(self, argument_name):
    """Get all the checks for this category.

        Find all globally visible functions where the first argument name
        starts with argument_name and which contain selected tests.
        """
    checks = []
    for check, attrs in _checks[argument_name].items():
        codes, args = attrs
        if any((not (code and self.ignore_code(code)) for code in codes)):
            checks.append((check.__name__, check, args))
    return sorted(checks)