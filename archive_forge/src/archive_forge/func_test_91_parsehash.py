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
def test_91_parsehash(self):
    """test parsehash()"""
    from passlib import hash
    result = hash.des_crypt.parsehash('OgAwTx2l6NADI')
    self.assertEqual(result, {'checksum': u('AwTx2l6NADI'), 'salt': u('Og')})
    h = '$5$LKO/Ute40T3FNF95$U0prpBQd4PloSGU0pnpM4z9wKn4vZ1.jsrzQfPqxph9'
    s = u('LKO/Ute40T3FNF95')
    c = u('U0prpBQd4PloSGU0pnpM4z9wKn4vZ1.jsrzQfPqxph9')
    result = hash.sha256_crypt.parsehash(h)
    self.assertEqual(result, dict(salt=s, rounds=5000, implicit_rounds=True, checksum=c))
    result = hash.sha256_crypt.parsehash(h, checksum=False)
    self.assertEqual(result, dict(salt=s, rounds=5000, implicit_rounds=True))
    result = hash.sha256_crypt.parsehash(h, sanitize=True)
    self.assertEqual(result, dict(rounds=5000, implicit_rounds=True, salt=u('LK**************'), checksum=u('U0pr***************************************')))
    result = hash.sha256_crypt.parsehash('$5$rounds=10428$uy/jIAhCetNCTtb0$YWvUOXbkqlqhyoPMpN8BMe.ZGsGx2aBvxTvDFI613c3')
    self.assertEqual(result, dict(checksum=u('YWvUOXbkqlqhyoPMpN8BMe.ZGsGx2aBvxTvDFI613c3'), salt=u('uy/jIAhCetNCTtb0'), rounds=10428))
    h1 = '$pbkdf2$60000$DoEwpvQeA8B4T.k951yLUQ$O26Y3/NJEiLCVaOVPxGXshyjW8k'
    result = hash.pbkdf2_sha1.parsehash(h1)
    self.assertEqual(result, dict(checksum=b';n\x98\xdf\xf3I\x12"\xc2U\xa3\x95?\x11\x97\xb2\x1c\xa3[\xc9', rounds=60000, salt=b'\x0e\x810\xa6\xf4\x1e\x03\xc0xO\xe9=\xe7\\\x8bQ'))
    result = hash.pbkdf2_sha1.parsehash(h1, sanitize=True)
    self.assertEqual(result, dict(checksum=u('O26************************'), rounds=60000, salt=u('Do********************')))