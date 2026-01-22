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
def select_by_size(self, num):
    """Select snapshots for minimum output size"""
    num -= len(self._snapshots)
    new_snapshots = self.get_size_ranking()[-num:]
    return [v for n, v in new_snapshots]