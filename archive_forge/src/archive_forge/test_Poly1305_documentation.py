import json
import unittest
from binascii import unhexlify, hexlify
from .common import make_mac_tests
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import Poly1305
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
Self-test suite for Cryptodome.Hash._Poly1305