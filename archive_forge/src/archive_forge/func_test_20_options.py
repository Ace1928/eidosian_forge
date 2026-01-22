from __future__ import with_statement
from passlib.utils.compat import PY3
import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, LazyCryptContext
from passlib.exc import PasslibConfigWarning, PasslibHashWarning
from passlib.utils import tick, to_unicode
from passlib.utils.compat import irange, u, unicode, str_to_uascii, PY2, PY26
import passlib.utils.handlers as uh
from passlib.tests.utils import (TestCase, set_file, TICK_RESOLUTION,
from passlib.registry import (register_crypt_handler_path,
import hashlib, time
def test_20_options(self):
    """test basic option parsing"""

    def parse(**kwds):
        return CryptContext(**kwds).to_dict()
    self.assertRaises(TypeError, CryptContext, __=0.1)
    self.assertRaises(TypeError, CryptContext, default__scheme__='x')
    self.assertRaises(TypeError, CryptContext, __option='x')
    self.assertRaises(TypeError, CryptContext, default____option='x')
    self.assertRaises(TypeError, CryptContext, __scheme__option='x')
    self.assertRaises(TypeError, CryptContext, category__scheme__option__invalid=30000)
    self.assertRaises(KeyError, parse, **{'admin.context__schemes': 'md5_crypt'})
    ctx = CryptContext(**{'schemes': 'md5_crypt,des_crypt', 'admin.context__default': 'des_crypt'})
    self.assertEqual(ctx.default_scheme('admin'), 'des_crypt')
    result = dict(default='md5_crypt')
    self.assertEqual(parse(default='md5_crypt'), result)
    self.assertEqual(parse(context__default='md5_crypt'), result)
    self.assertEqual(parse(default__context__default='md5_crypt'), result)
    self.assertEqual(parse(**{'context.default': 'md5_crypt'}), result)
    self.assertEqual(parse(**{'default.context.default': 'md5_crypt'}), result)
    result = dict(admin__context__default='md5_crypt')
    self.assertEqual(parse(admin__context__default='md5_crypt'), result)
    self.assertEqual(parse(**{'admin.context.default': 'md5_crypt'}), result)
    result = dict(all__vary_rounds=0.1)
    self.assertEqual(parse(all__vary_rounds=0.1), result)
    self.assertEqual(parse(default__all__vary_rounds=0.1), result)
    self.assertEqual(parse(**{'all.vary_rounds': 0.1}), result)
    self.assertEqual(parse(**{'default.all.vary_rounds': 0.1}), result)
    result = dict(admin__all__vary_rounds=0.1)
    self.assertEqual(parse(admin__all__vary_rounds=0.1), result)
    self.assertEqual(parse(**{'admin.all.vary_rounds': 0.1}), result)
    ctx = CryptContext(['phpass', 'md5_crypt'], phpass__ident='P')
    self.assertRaises(KeyError, ctx.copy, md5_crypt__ident='P')
    self.assertRaises(KeyError, CryptContext, schemes=['des_crypt'], des_crypt__salt='xx')
    self.assertRaises(KeyError, CryptContext, schemes=['des_crypt'], all__salt='xx')