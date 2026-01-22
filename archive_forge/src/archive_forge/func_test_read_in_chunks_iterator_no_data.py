import sys
import random
import socket
import string
import os.path
import platform
import unittest
import warnings
from itertools import chain
import pytest
import libcloud.utils.files
from libcloud.utils.py3 import StringIO, b, bchr, urlquote, hexadigits
from libcloud.utils.misc import get_driver, set_driver, get_secure_random_string
from libcloud.common.types import LibcloudError
from libcloud.compute.types import Provider
from libcloud.utils.publickey import get_pubkey_ssh2_fingerprint, get_pubkey_openssh_fingerprint
from libcloud.utils.decorators import wrap_non_libcloud_exceptions
from libcloud.utils.networking import (
from libcloud.compute.providers import DRIVERS
from libcloud.compute.drivers.dummy import DummyNodeDriver
from libcloud.storage.drivers.dummy import DummyIterator
from io import FileIO as file
def test_read_in_chunks_iterator_no_data(self):
    iterator = DummyIterator()
    generator1 = libcloud.utils.files.read_in_chunks(iterator=iterator, yield_empty=False)
    generator2 = libcloud.utils.files.read_in_chunks(iterator=iterator, yield_empty=True)
    count = 0
    for data in generator1:
        count += 1
        self.assertEqual(data, b(''))
    self.assertEqual(count, 0)
    count = 0
    for data in generator2:
        count += 1
        self.assertEqual(data, b(''))
    self.assertEqual(count, 1)