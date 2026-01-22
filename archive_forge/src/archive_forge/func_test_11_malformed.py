from __future__ import with_statement
from logging import getLogger
import os
import subprocess
from passlib import apache, registry
from passlib.exc import MissingBackendError
from passlib.utils.compat import irange
from passlib.tests.backports import unittest
from passlib.tests.utils import TestCase, get_file, set_file, ensure_mtime_changed
from passlib.utils.compat import u
from passlib.utils import to_bytes
from passlib.utils.handlers import to_unicode_for_identify
def test_11_malformed(self):
    self.assertRaises(ValueError, apache.HtdigestFile.from_string, b'realm:user1:pass1:other\n')
    self.assertRaises(ValueError, apache.HtdigestFile.from_string, b'user1:pass1\n')