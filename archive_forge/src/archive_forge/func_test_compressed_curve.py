import os
import errno
import warnings
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import bord, tostr, FileNotFoundError
from Cryptodome.Util.asn1 import DerSequence, DerBitString
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Hash import SHAKE128
from Cryptodome.PublicKey import ECC
def test_compressed_curve(self):
    pem1 = '-----BEGIN EC PRIVATE KEY-----\nMIHcAgEBBEIAnm1CEjVjvNfXEN730p+D6su5l+mOztdc5XmTEoti+s2R4GQ4mAv3\n0zYLvyklvOHw0+yy8d0cyGEJGb8T3ZVKmg2gBwYFK4EEACOhgYkDgYYABAHzjTI1\nckxQ3Togi0LAxiG0PucdBBBs5oIy3df95xv6SInp70z+4qQ2EltEmdNMssH8eOrl\nM5CYdZ6nbcHMVaJUvQEzTrYxvFjOgJiOd+E9eBWbLkbMNqsh1UKVO6HbMbW0ohCI\nuGxO8tM6r3w89/qzpG2SvFM/fvv3mIR30wSZDD84qA==\n-----END EC PRIVATE KEY-----'
    pem2 = '-----BEGIN EC PRIVATE KEY-----\nMIHcAgEBBEIB84OfhJluLBRLn3+cC/RQ37C2SfQVP/t0gQK2tCsTf5avRcWYRrOJ\nPmX9lNnkC0Hobd75QFRmdxrB0Wd1/M4jZOWgBwYFK4EEACOhgYkDgYYABAAMZcdJ\n1YLCGHt3bHCEzdidVy6+brlJIbv1aQ9fPQLF7WKNv4c8w3H8d5a2+SDZilBOsk5c\n6cNJDMz2ExWQvxl4CwDJtJGt1+LHVKFGy73NANqVxMbRu+2F8lOxkNp/ziFTbVyV\nvv6oYkMIIi7r5oQWAiQDrR2mlrrFDL9V7GH/r8SWQw==\n-----END EC PRIVATE KEY-----'
    key1 = ECC.import_key(pem1)
    low16 = int(key1.pointQ.y % 65536)
    self.assertEqual(low16, 14504)
    key2 = ECC.import_key(pem2)
    low16 = int(key2.pointQ.y % 65536)
    self.assertEqual(low16, 38467)