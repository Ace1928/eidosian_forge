import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
def test_get_auth_default_no_feature(self):
    n = NodeDriver('foo')
    self.assertEqual(None, n._get_and_check_auth(None))