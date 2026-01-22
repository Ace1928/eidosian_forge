from __future__ import unicode_literals
from builtins import str
from past.builtins import basestring
import hashlib
import sys
def get_hash_int(item):
    return int(get_hash(item), base=16)