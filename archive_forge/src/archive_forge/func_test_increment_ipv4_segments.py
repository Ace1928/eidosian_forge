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
def test_increment_ipv4_segments(self):
    values = [(('127', '0', '0', '1'), '127.0.0.2'), (('255', '255', '255', '0'), '255.255.255.1'), (('254', '255', '255', '255'), '255.0.0.0'), (('100', '1', '0', '255'), '100.1.1.0')]
    for segments, incremented_ip in values:
        result = increment_ipv4_segments(segments=segments)
        result = join_ipv4_segments(segments=result)
        self.assertEqual(result, incremented_ip)