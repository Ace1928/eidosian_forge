import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_zone_prices(self):
    location = NodeLocation(id='fi-hel1', name='Helsinki #1', country='FI', driver=None)
    self.assertEqual(self.pp.get_price('1xCPU-1GB', location), 1.588)