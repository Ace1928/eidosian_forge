from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_thin_pool_provisioned_capacity(self):
    self.vg.vg_thin_pool = 'test-prov-cap-pool-unit'
    self.vg.vg_name = 'test-prov-cap-vg-unit'
    self.assertIsNone(self.vg.create_thin_pool(name=self.vg.vg_thin_pool))
    self.assertEqual(9.5, self.vg.vg_thin_pool_size)
    self.assertEqual(7.6, self.vg.vg_thin_pool_free_space)
    self.assertEqual(3.0, self.vg.vg_provisioned_capacity)
    self.vg.vg_thin_pool = 'test-prov-cap-pool-no-unit'
    self.vg.vg_name = 'test-prov-cap-vg-no-unit'
    self.assertIsNone(self.vg.create_thin_pool(name=self.vg.vg_thin_pool))
    self.assertEqual(9.5, self.vg.vg_thin_pool_size)
    self.assertEqual(7.6, self.vg.vg_thin_pool_free_space)
    self.assertEqual(3.0, self.vg.vg_provisioned_capacity)