from __future__ import unicode_literals
import re
import pybtex.io
from pybtex.bibtex.interpreter import (
from pybtex.scanner import (
def process_int_literal(value):
    return Integer(int(value.strip('#')))