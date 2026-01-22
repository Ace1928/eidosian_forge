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
def test_encoded_length(self):
    su1 = ShortUUID()
    self.assertEqual(su1.encoded_length(), 22)
    base64_alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + '+/'
    su2 = ShortUUID(base64_alphabet)
    self.assertEqual(su2.encoded_length(), 22)
    binary_alphabet = '01'
    su3 = ShortUUID(binary_alphabet)
    self.assertEqual(su3.encoded_length(), 128)
    su4 = ShortUUID()
    self.assertEqual(su4.encoded_length(num_bytes=8), 11)