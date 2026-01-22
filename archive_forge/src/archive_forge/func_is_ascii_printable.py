import os
import string
import pytest
from .util import random_string
from keyring import errors
def is_ascii_printable(s):
    return all((32 <= ord(c) < 127 for c in s))