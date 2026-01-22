import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_plan_not_found_in_zone(self):
    location = NodeLocation(id='no_such_location', name='', country='', driver=None)
    self.assertIsNone(self.pp.get_price('1xCPU-1GB', location))