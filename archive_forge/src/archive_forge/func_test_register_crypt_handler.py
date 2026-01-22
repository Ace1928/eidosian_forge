from __future__ import with_statement
from logging import getLogger
import warnings
import sys
from passlib import hash, registry, exc
from passlib.registry import register_crypt_handler, register_crypt_handler_path, \
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase
def test_register_crypt_handler(self):
    """test register_crypt_handler()"""
    self.assertRaises(TypeError, register_crypt_handler, {})
    self.assertRaises(ValueError, register_crypt_handler, type('x', (uh.StaticHandler,), dict(name=None)))
    self.assertRaises(ValueError, register_crypt_handler, type('x', (uh.StaticHandler,), dict(name='AB_CD')))
    self.assertRaises(ValueError, register_crypt_handler, type('x', (uh.StaticHandler,), dict(name='ab-cd')))
    self.assertRaises(ValueError, register_crypt_handler, type('x', (uh.StaticHandler,), dict(name='ab__cd')))
    self.assertRaises(ValueError, register_crypt_handler, type('x', (uh.StaticHandler,), dict(name='default')))

    class dummy_1(uh.StaticHandler):
        name = 'dummy_1'

    class dummy_1b(uh.StaticHandler):
        name = 'dummy_1'
    self.assertTrue('dummy_1' not in list_crypt_handlers())
    register_crypt_handler(dummy_1)
    register_crypt_handler(dummy_1)
    self.assertIs(get_crypt_handler('dummy_1'), dummy_1)
    self.assertRaises(KeyError, register_crypt_handler, dummy_1b)
    self.assertIs(get_crypt_handler('dummy_1'), dummy_1)
    register_crypt_handler(dummy_1b, force=True)
    self.assertIs(get_crypt_handler('dummy_1'), dummy_1b)
    self.assertTrue('dummy_1' in list_crypt_handlers())