from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import hosts, hash as hashmod
from passlib.utils import unix_crypt_schemes
from passlib.tests.utils import TestCase
def test_bsd_contexts(self):
    for ctx in [hosts.freebsd_context, hosts.openbsd_context, hosts.netbsd_context]:
        for hash in ['$1$TXl/FX/U$BZge.lr.ux6ekjEjxmzwz0', 'kAJJz.Rwp0A/I']:
            self.assertTrue(ctx.verify('test', hash))
        h1 = '$2a$04$yjDgE74RJkeqC0/1NheSSOrvKeu9IbKDpcQf/Ox3qsrRS/Kw42qIS'
        if hashmod.bcrypt.has_backend():
            self.assertTrue(ctx.verify('test', h1))
        else:
            self.assertEqual(ctx.identify(h1), 'bcrypt')
        self.check_unix_disabled(ctx)