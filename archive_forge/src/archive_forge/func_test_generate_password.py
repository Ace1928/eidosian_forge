from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_generate_password(self):
    """generate_password()"""
    from passlib.utils import generate_password
    warnings.filterwarnings('ignore', 'The function.*generate_password\\(\\) is deprecated')
    self.assertEqual(len(generate_password(15)), 15)