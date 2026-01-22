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
def test_24_vary_rounds(self):
    """test 'vary_rounds' hash option parsing"""

    def parse(v):
        return CryptContext(all__vary_rounds=v).to_dict()['all__vary_rounds']
    self.assertEqual(parse(0.1), 0.1)
    self.assertEqual(parse('0.1'), 0.1)
    self.assertEqual(parse('10%'), 0.1)
    self.assertEqual(parse(1000), 1000)
    self.assertEqual(parse('1000'), 1000)