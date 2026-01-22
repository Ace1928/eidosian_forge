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
@requires_htpasswd_cmd
@unittest.skipUnless(registry.has_backend('bcrypt'), 'bcrypt support required')
def test_htpasswd_cmd_verify_bcrypt(self):
    """
        verify "htpasswd" command can read bcrypt format

        this tests for regression of issue 95, where we output "$2b$" instead of "$2y$";
        fixed in v1.7.2.
        """
    path = self.mktemp()
    ht = apache.HtpasswdFile(path=path, new=True)

    def hash_scheme(pwd, scheme):
        return ht.context.handler(scheme).hash(pwd)
    ht.set_hash('user1', hash_scheme('password', 'bcrypt'))
    ht.save()
    self.assertFalse(_call_htpasswd_verify(path, 'user1', 'wrong'))
    if HAVE_HTPASSWD_BCRYPT:
        self.assertTrue(_call_htpasswd_verify(path, 'user1', 'password'))
    else:
        self.assertFalse(_call_htpasswd_verify(path, 'user1', 'password'))