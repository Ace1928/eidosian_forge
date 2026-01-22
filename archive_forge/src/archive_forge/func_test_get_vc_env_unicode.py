import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
def test_get_vc_env_unicode(self):
    import distutils._msvccompiler as _msvccompiler
    test_var = 'ṰḖṤṪ┅ṼẨṜ'
    test_value = '₃⁴₅'
    old_distutils_use_sdk = os.environ.pop('DISTUTILS_USE_SDK', None)
    os.environ[test_var] = test_value
    try:
        env = _msvccompiler._get_vc_env('x86')
        self.assertIn(test_var.lower(), env)
        self.assertEqual(test_value, env[test_var.lower()])
    finally:
        os.environ.pop(test_var)
        if old_distutils_use_sdk:
            os.environ['DISTUTILS_USE_SDK'] = old_distutils_use_sdk