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
def write_installed_files(self, paths, prefix, dry_run=False):
    """
        Writes the ``RECORD`` file, using the ``paths`` iterable passed in. Any
        existing ``RECORD`` file is silently overwritten.

        prefix is used to determine when to write absolute paths.
        """
    prefix = os.path.join(prefix, '')
    base = os.path.dirname(self.path)
    base_under_prefix = base.startswith(prefix)
    base = os.path.join(base, '')
    record_path = self.get_distinfo_file('RECORD')
    logger.info('creating %s', record_path)
    if dry_run:
        return None
    with CSVWriter(record_path) as writer:
        for path in paths:
            if os.path.isdir(path) or path.endswith(('.pyc', '.pyo')):
                hash_value = size = ''
            else:
                size = '%d' % os.path.getsize(path)
                with open(path, 'rb') as fp:
                    hash_value = self.get_hash(fp.read())
            if path.startswith(base) or (base_under_prefix and path.startswith(prefix)):
                path = os.path.relpath(path, base)
            writer.writerow((path, hash_value, size))
        if record_path.startswith(base):
            record_path = os.path.relpath(record_path, base)
        writer.writerow((record_path, '', ''))
    return record_path