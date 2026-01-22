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
def cache_version(self, version_id):
    try:
        return self._lines[version_id]
    except KeyError:
        pass
    diff = self.get_diff(version_id)
    lines = []
    reconstructor = _Reconstructor(self, self._lines, self._parents)
    reconstructor.reconstruct_version(lines, version_id)
    self._lines[version_id] = lines
    return lines