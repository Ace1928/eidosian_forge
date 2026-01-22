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
def test_09_encodings(self):
    """test encoding parameter"""
    self.assertRaises(ValueError, apache.HtdigestFile, encoding='utf-16')
    ht = apache.HtdigestFile.from_string(self.sample_04_utf8, encoding='utf-8', return_unicode=True)
    self.assertEqual(ht.realms(), [u('realmæ')])
    self.assertEqual(ht.users(u('realmæ')), [u('useræ')])
    ht = apache.HtdigestFile.from_string(self.sample_04_latin1, encoding='latin-1', return_unicode=True)
    self.assertEqual(ht.realms(), [u('realmæ')])
    self.assertEqual(ht.users(u('realmæ')), [u('useræ')])