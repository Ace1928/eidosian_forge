import os
import string
import sys
import unittest
from collections import defaultdict
from unittest.mock import patch
from uuid import UUID
from uuid import uuid4
from shortuuid.cli import cli
from shortuuid.main import decode
from shortuuid.main import encode
from shortuuid.main import get_alphabet
from shortuuid.main import random
from shortuuid.main import set_alphabet
from shortuuid.main import ShortUUID
from shortuuid.main import uuid
def test_decode_dict(self):
    su = ShortUUID()
    self.assertRaises(ValueError, su.decode, [])
    self.assertRaises(ValueError, su.decode, {})
    self.assertRaises(ValueError, su.decode, (2,))
    self.assertRaises(ValueError, su.decode, 42)
    self.assertRaises(ValueError, su.decode, 42.0)