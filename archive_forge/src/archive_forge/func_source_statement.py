import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
@property
def source_statement(self):
    return 'from ' + self.fullName + ' import *'