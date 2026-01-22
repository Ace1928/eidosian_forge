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
def test_02_set_password_autosave(self):
    path = self.mktemp()
    sample = b'user1:pass1\n'
    set_file(path, sample)
    ht = apache.HtpasswdFile(path)
    ht.set_password('user1', 'pass2')
    self.assertEqual(get_file(path), sample)
    ht = apache.HtpasswdFile(path, default_scheme='plaintext', autosave=True)
    ht.set_password('user1', 'pass2')
    self.assertEqual(get_file(path), b'user1:pass2\n')