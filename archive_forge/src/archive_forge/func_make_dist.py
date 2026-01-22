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
def make_dist(name, version, **kwargs):
    """
    A convenience method for making a dist given just a name and version.
    """
    summary = kwargs.pop('summary', 'Placeholder for summary')
    md = Metadata(**kwargs)
    md.name = name
    md.version = version
    md.summary = summary or 'Placeholder for summary'
    return Distribution(md)