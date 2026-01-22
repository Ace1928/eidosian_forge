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
def test_07_realms(self):
    """test realms() & delete_realm()"""
    ht = apache.HtdigestFile.from_string(self.sample_01)
    self.assertEqual(ht.delete_realm('x'), 0)
    self.assertEqual(ht.realms(), ['realm'])
    self.assertEqual(ht.delete_realm('realm'), 4)
    self.assertEqual(ht.realms(), [])
    self.assertEqual(ht.to_string(), b'')