from __future__ import with_statement
from logging import getLogger
import warnings
import sys
from passlib import hash, registry, exc
from passlib.registry import register_crypt_handler, register_crypt_handler_path, \
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase
def test_register_crypt_handler_path(self):
    """test register_crypt_handler_path()"""
    paths = registry._locations
    self.assertTrue('dummy_0' not in paths)
    self.assertFalse(hasattr(hash, 'dummy_0'))
    self.assertRaises(ValueError, register_crypt_handler_path, 'dummy_0', '.test_registry')
    self.assertRaises(ValueError, register_crypt_handler_path, 'dummy_0', __name__ + ':dummy_0:xxx')
    self.assertRaises(ValueError, register_crypt_handler_path, 'dummy_0', __name__ + ':dummy_0.xxx')
    register_crypt_handler_path('dummy_0', __name__)
    self.assertTrue('dummy_0' in list_crypt_handlers())
    self.assertTrue('dummy_0' not in list_crypt_handlers(loaded_only=True))
    self.assertIs(hash.dummy_0, dummy_0)
    self.assertTrue('dummy_0' in list_crypt_handlers(loaded_only=True))
    unload_handler_name('dummy_0')
    register_crypt_handler_path('dummy_0', __name__ + ':alt_dummy_0')
    self.assertIs(hash.dummy_0, alt_dummy_0)
    unload_handler_name('dummy_0')
    register_crypt_handler_path('dummy_x', __name__)
    self.assertRaises(TypeError, get_crypt_handler, 'dummy_x')
    register_crypt_handler_path('alt_dummy_0', __name__)
    self.assertRaises(ValueError, get_crypt_handler, 'alt_dummy_0')
    unload_handler_name('alt_dummy_0')
    sys.modules.pop('passlib.tests._test_bad_register', None)
    register_crypt_handler_path('dummy_bad', 'passlib.tests._test_bad_register')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'xxxxxxxxxx', DeprecationWarning)
        h = get_crypt_handler('dummy_bad')
    from passlib.tests import _test_bad_register as tbr
    self.assertIs(h, tbr.alt_dummy_bad)