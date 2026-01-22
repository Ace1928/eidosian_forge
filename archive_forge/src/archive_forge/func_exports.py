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
@cached_property
def exports(self):
    """
        Return the information exported by this distribution.
        :return: A dictionary of exports, mapping an export category to a dict
                 of :class:`ExportEntry` instances describing the individual
                 export entries, and keyed by name.
        """
    result = {}
    r = self.get_distinfo_resource(EXPORTS_FILENAME)
    if r:
        result = self.read_exports()
    return result