import os
import sys
import libcloud.security
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeAuthPassword
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
def test_list_nodes_returned_successfully(self):
    vmimages = self.driver.list_nodes(ex_cloud_service_name='dcoddkinztest01')
    self.assertEqual(len(vmimages), 2)
    img0 = vmimages[0]
    self.assertEqual(img0.id, 'dc03')
    self.assertEqual(img0.name, 'dc03')
    self.assertListEqual(img0.public_ips, ['191.235.135.62'])
    self.assertListEqual(img0.private_ips, ['100.92.66.69'])
    self.assertIsNone(img0.size)
    self.assertEqual(img0.state, NodeState.RUNNING)
    self.assertTrue(isinstance(img0.extra, dict))
    extra = img0.extra
    self.assertEqual(extra['instance_size'], 'Small')
    self.assertEqual(extra['power_state'], 'Started')
    self.assertEqual(extra['ssh_port'], '22')