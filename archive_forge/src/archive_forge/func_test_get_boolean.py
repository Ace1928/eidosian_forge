import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_get_boolean(self):
    cd = ConfigDict()
    cd.set((b'core',), b'foo', b'true')
    self.assertTrue(cd.get_boolean((b'core',), b'foo'))
    cd.set((b'core',), b'foo', b'false')
    self.assertFalse(cd.get_boolean((b'core',), b'foo'))
    cd.set((b'core',), b'foo', b'invalid')
    self.assertRaises(ValueError, cd.get_boolean, (b'core',), b'foo')