import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_from_file_utf8_bom(self):
    text = '[core]\nfoo = bär\n'.encode('utf-8-sig')
    cf = self.from_file(text)
    self.assertEqual(b'b\xc3\xa4r', cf.get((b'core',), b'foo'))