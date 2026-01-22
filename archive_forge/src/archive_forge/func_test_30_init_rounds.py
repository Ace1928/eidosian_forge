from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
def test_30_init_rounds(self):
    """test GenericHandler + HasRounds mixin"""

    class d1(uh.HasRounds, uh.GenericHandler):
        name = 'd1'
        setting_kwds = ('rounds',)
        min_rounds = 1
        max_rounds = 3
        default_rounds = 2

    def norm_rounds(**k):
        return d1(**k).rounds
    self.assertRaises(TypeError, norm_rounds)
    self.assertRaises(TypeError, norm_rounds, rounds=None)
    self.assertEqual(norm_rounds(use_defaults=True), 2)
    self.assertRaises(TypeError, norm_rounds, rounds=1.5)
    with warnings.catch_warnings(record=True) as wlog:
        self.assertRaises(ValueError, norm_rounds, rounds=0)
        self.consumeWarningList(wlog)
        self.assertEqual(norm_rounds(rounds=1), 1)
        self.assertEqual(norm_rounds(rounds=2), 2)
        self.assertEqual(norm_rounds(rounds=3), 3)
        self.consumeWarningList(wlog)
        self.assertRaises(ValueError, norm_rounds, rounds=4)
        self.consumeWarningList(wlog)
    d1.default_rounds = None
    self.assertRaises(TypeError, norm_rounds, use_defaults=True)