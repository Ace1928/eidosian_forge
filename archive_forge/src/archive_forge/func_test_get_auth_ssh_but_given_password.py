import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
def test_get_auth_ssh_but_given_password(self):
    n = NodeDriver('foo')
    n.features = {'create_node': ['ssh_key']}
    auth = NodeAuthPassword('password')
    self.assertRaises(LibcloudError, n._get_and_check_auth, auth)