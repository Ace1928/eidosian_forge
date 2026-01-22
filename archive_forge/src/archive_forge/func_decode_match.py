from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def decode_match(match):
    return utf8_bytes_string(codecs.decode(match.group(0), 'unicode-escape'))