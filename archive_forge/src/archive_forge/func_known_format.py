import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def known_format(self):
    """Returns True iff the format is known."""
    return self.format in _KNOWN_FORMATS