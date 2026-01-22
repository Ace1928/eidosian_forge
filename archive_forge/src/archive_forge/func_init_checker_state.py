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
def init_checker_state(self, name, argument_names):
    """Prepare custom state for the specific checker plugin."""
    if 'checker_state' in argument_names:
        self.checker_state = self._checker_states.setdefault(name, {})