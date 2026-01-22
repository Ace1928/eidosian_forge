from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
def get_distinfo_resource(self, path):
    if path not in DIST_FILES:
        raise DistlibException('invalid path for a dist-info file: %r at %r' % (path, self.path))
    finder = resources.finder_for_path(self.path)
    if finder is None:
        raise DistlibException('Unable to get a finder for %s' % self.path)
    return finder.find(path)