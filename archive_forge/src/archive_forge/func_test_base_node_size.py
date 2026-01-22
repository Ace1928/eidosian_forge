import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
def test_base_node_size(self):
    NodeSize(id=0, name=0, ram=0, disk=0, bandwidth=0, price=0, driver=FakeDriver())