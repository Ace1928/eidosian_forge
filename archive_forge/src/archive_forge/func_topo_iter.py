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
def topo_iter(vf, versions=None):
    if versions is None:
        versions = vf.versions()
    parents = vf.get_parent_map(versions)
    return _topo_iter(parents, versions)