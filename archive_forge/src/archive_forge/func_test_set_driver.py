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
def test_set_driver(self):
    try:
        driver = set_driver(DRIVERS, Provider.DUMMY, 'libcloud.storage.drivers.dummy', 'DummyStorageDriver')
    except AttributeError:
        pass
    driver = set_driver(DRIVERS, 'testingset', 'libcloud.storage.drivers.dummy', 'DummyStorageDriver')
    self.assertTrue(driver is not None)
    try:
        set_driver(DRIVERS, 'testingset', 'libcloud.storage.drivers.dummy', 'DummyStorageDriver')
    except AttributeError:
        pass
    try:
        set_driver(DRIVERS, 'testingnew', 'libcloud.storage.drivers.dummy1', 'DummyStorageDriver')
    except ImportError:
        pass
    try:
        set_driver(DRIVERS, 'testingnew', 'libcloud.storage.drivers.dummy', 'DummyStorageDriver1')
    except AttributeError:
        pass