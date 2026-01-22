import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
def test_ex_node_set_save_name(self):
    """
        Test ex_node_action functionality.
        """
    image = NodeImage(id=5, name='Ubuntu 9.04 LAMP', driver=self.driver)
    node = Node(5, None, None, None, None, self.driver, image=image)
    ret = self.driver.ex_node_set_save_name(node, 'test')
    self.assertTrue(ret)