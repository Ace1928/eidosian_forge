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
def test_50_norm_ident(self):
    """test GenericHandler + HasManyIdents"""

    class d1(uh.HasManyIdents, uh.GenericHandler):
        name = 'd1'
        setting_kwds = ('ident',)
        default_ident = u('!A')
        ident_values = (u('!A'), u('!B'))
        ident_aliases = {u('A'): u('!A')}

    def norm_ident(**k):
        return d1(**k).ident
    self.assertRaises(TypeError, norm_ident)
    self.assertRaises(TypeError, norm_ident, ident=None)
    self.assertEqual(norm_ident(use_defaults=True), u('!A'))
    self.assertEqual(norm_ident(ident=u('!A')), u('!A'))
    self.assertEqual(norm_ident(ident=u('!B')), u('!B'))
    self.assertRaises(ValueError, norm_ident, ident=u('!C'))
    self.assertEqual(norm_ident(ident=u('A')), u('!A'))
    self.assertRaises(ValueError, norm_ident, ident=u('B'))
    self.assertTrue(d1.identify(u('!Axxx')))
    self.assertTrue(d1.identify(u('!Bxxx')))
    self.assertFalse(d1.identify(u('!Cxxx')))
    self.assertFalse(d1.identify(u('A')))
    self.assertFalse(d1.identify(u('')))
    self.assertRaises(TypeError, d1.identify, None)
    self.assertRaises(TypeError, d1.identify, 1)
    d1.default_ident = None
    self.assertRaises(AssertionError, norm_ident, use_defaults=True)