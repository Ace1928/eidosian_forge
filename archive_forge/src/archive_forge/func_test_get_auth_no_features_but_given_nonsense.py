import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
def test_get_auth_no_features_but_given_nonsense(self):
    n = NodeDriver('foo')
    auth = 'nonsense'
    self.assertRaises(LibcloudError, n._get_and_check_auth, auth)