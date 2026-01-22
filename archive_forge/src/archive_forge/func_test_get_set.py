import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_get_set(self):
    cd = ConfigDict()
    self.assertRaises(KeyError, cd.get, b'foo', b'core')
    cd.set((b'core',), b'foo', b'bla')
    self.assertEqual(b'bla', cd.get((b'core',), b'foo'))
    cd.set((b'core',), b'foo', b'bloe')
    self.assertEqual(b'bloe', cd.get((b'core',), b'foo'))