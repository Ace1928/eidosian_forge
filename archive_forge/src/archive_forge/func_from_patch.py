import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
@classmethod
def from_patch(cls, text):
    """Create a MultiParent from its string form"""
    return cls._from_patch(BytesIO(text))