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
def get_size_ranking(self):
    """Get versions ranked by size"""
    versions = []
    for version_id in self.versions():
        if version_id in self._snapshots:
            continue
        diff_len = self.get_diff(version_id).patch_len()
        snapshot_len = MultiParent([NewText(self.cache_version(version_id))]).patch_len()
        versions.append((snapshot_len - diff_len, version_id))
    versions.sort()
    return versions