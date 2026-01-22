import sys
import string
import unittest
from unittest.mock import Mock, patch
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.nfsn import NFSNConnection
def test_salt_characters(self):
    """salt must be alphanumeric"""
    salt_characters = string.ascii_letters + string.digits
    for c in self.driver._salt():
        self.assertIn(c, salt_characters)