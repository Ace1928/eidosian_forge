from __future__ import with_statement
from logging import getLogger
import warnings
import sys
from passlib import hash, registry, exc
from passlib.registry import register_crypt_handler, register_crypt_handler_path, \
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase
def test_list_crypt_handlers(self):
    """test list_crypt_handlers()"""
    from passlib.registry import list_crypt_handlers
    hash.__dict__['_fake'] = 'dummy'
    for name in list_crypt_handlers():
        self.assertFalse(name.startswith('_'), '%r: ' % name)
    unload_handler_name('_fake')