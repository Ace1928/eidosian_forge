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
def import_diffs(self, vf):
    """Import the diffs from another pseudo-versionedfile"""
    for version_id in vf.versions():
        self.add_diff(vf.get_diff(version_id), version_id, vf._parents[version_id])