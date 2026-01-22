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
def test_alphabet(self):
    alphabet = '01'
    su1 = ShortUUID(alphabet)
    su2 = ShortUUID()
    self.assertEqual(alphabet, su1.get_alphabet())
    su1.set_alphabet('01010101010101')
    self.assertEqual(alphabet, su1.get_alphabet())
    self.assertEqual(set(su1.uuid()), set('01'))
    self.assertTrue(116 < len(su1.uuid()) < 140)
    self.assertTrue(20 < len(su2.uuid()) < 24)
    u = uuid4()
    self.assertEqual(u, su1.decode(su1.encode(u)))
    u = su1.uuid()
    self.assertEqual(u, su1.encode(su1.decode(u)))
    self.assertRaises(ValueError, su1.set_alphabet, '1')
    self.assertRaises(ValueError, su1.set_alphabet, '1111111')