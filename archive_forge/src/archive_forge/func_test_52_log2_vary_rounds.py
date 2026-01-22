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
def test_52_log2_vary_rounds(self):
    """test log2 vary rounds"""
    cc = CryptContext(schemes=['bcrypt'], bcrypt__min_rounds=15, bcrypt__max_rounds=25, bcrypt__default_rounds=20)
    self.assertRaises(ValueError, cc.copy, all__vary_rounds=-1)
    self.assertRaises(ValueError, cc.copy, all__vary_rounds='-1%')
    self.assertRaises(ValueError, cc.copy, all__vary_rounds='101%')
    c2 = cc.copy(all__vary_rounds=0)
    self.assertEqual(c2._get_record('bcrypt', None).vary_rounds, 0)
    self.assert_rounds_range(c2, 'bcrypt', 20, 20)
    c2 = cc.copy(all__vary_rounds='0%')
    self.assertEqual(c2._get_record('bcrypt', None).vary_rounds, 0)
    self.assert_rounds_range(c2, 'bcrypt', 20, 20)
    c2 = cc.copy(all__vary_rounds=1)
    self.assertEqual(c2._get_record('bcrypt', None).vary_rounds, 1)
    self.assert_rounds_range(c2, 'bcrypt', 19, 21)
    c2 = cc.copy(all__vary_rounds=100)
    self.assertEqual(c2._get_record('bcrypt', None).vary_rounds, 100)
    self.assert_rounds_range(c2, 'bcrypt', 15, 25)
    c2 = cc.copy(all__vary_rounds='1%')
    self.assertEqual(c2._get_record('bcrypt', None).vary_rounds, 0.01)
    self.assert_rounds_range(c2, 'bcrypt', 20, 20)
    c2 = cc.copy(all__vary_rounds='49%')
    self.assertEqual(c2._get_record('bcrypt', None).vary_rounds, 0.49)
    self.assert_rounds_range(c2, 'bcrypt', 20, 20)
    c2 = cc.copy(all__vary_rounds='50%')
    self.assertEqual(c2._get_record('bcrypt', None).vary_rounds, 0.5)
    self.assert_rounds_range(c2, 'bcrypt', 19, 20)
    c2 = cc.copy(all__vary_rounds='100%')
    self.assertEqual(c2._get_record('bcrypt', None).vary_rounds, 1.0)
    self.assert_rounds_range(c2, 'bcrypt', 15, 21)