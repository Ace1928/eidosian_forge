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
def pending_parents(version):
    if parents[version] is None:
        return []
    return [v for v in parents[version] if v in versions and v not in seen]