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
def test_06_save(self):
    """test save()"""
    path = self.mktemp()
    set_file(path, self.sample_01)
    ht = apache.HtdigestFile(path)
    ht.delete('user1', 'realm')
    ht.delete('user2', 'realm')
    ht.save()
    self.assertEqual(get_file(path), self.sample_02)
    hb = apache.HtdigestFile()
    hb.set_password('user1', 'realm', 'pass1')
    self.assertRaises(RuntimeError, hb.save)
    hb.save(path)
    self.assertEqual(get_file(path), hb.to_string())