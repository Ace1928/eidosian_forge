import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_90_algs(self):
    """test parsing of 'algs' setting"""
    defaults = dict(salt=b'A' * 10, rounds=1000)

    def parse(algs, **kwds):
        for k in defaults:
            kwds.setdefault(k, defaults[k])
        return self.handler(algs=algs, **kwds).algs
    self.assertEqual(parse(None, use_defaults=True), hash.scram.default_algs)
    self.assertRaises(TypeError, parse, None)
    self.assertEqual(parse('sha1'), ['sha-1'])
    self.assertEqual(parse('sha1, sha256, md5'), ['md5', 'sha-1', 'sha-256'])
    self.assertEqual(parse(['sha-1', 'sha256']), ['sha-1', 'sha-256'])
    self.assertRaises(ValueError, parse, ['sha-256'])
    self.assertRaises(ValueError, parse, algs=[], use_defaults=True)
    self.assertRaises(ValueError, parse, ['sha-1', 'shaxxx-190'])
    self.assertRaises(RuntimeError, parse, ['sha-1'], checksum={'sha-1': b'\x00' * 20})