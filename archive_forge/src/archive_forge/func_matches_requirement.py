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
def matches_requirement(self, req):
    """
        Say if this instance matches (fulfills) a requirement.
        :param req: The requirement to match.
        :rtype req: str
        :return: True if it matches, else False.
        """
    r = parse_requirement(req)
    scheme = get_scheme(self.metadata.scheme)
    try:
        matcher = scheme.matcher(r.requirement)
    except UnsupportedVersionError:
        logger.warning('could not read version %r - using name only', req)
        name = req.split()[0]
        matcher = scheme.matcher(name)
    name = matcher.key
    result = False
    for p in self.provides:
        p_name, p_ver = parse_name_and_version(p)
        if p_name != name:
            continue
        try:
            result = matcher.match(p_ver)
            break
        except UnsupportedVersionError:
            pass
    return result