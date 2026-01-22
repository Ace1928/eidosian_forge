import unittest
from Cryptodome.Util.py3compat import bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import Salsa20
from .common import make_stream_tests
Verify we can encrypt or decrypt bytearrays