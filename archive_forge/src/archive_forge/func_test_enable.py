from unittest import mock
from openstack.block_storage.v3 import service
from openstack.tests.unit import base
def test_enable(self):
    sot = service.Service(**EXAMPLE)
    res = sot.enable(self.sess)
    self.assertIsNotNone(res)
    url = 'os-services/enable'
    body = {'binary': 'cinder-scheduler', 'host': 'devstack'}
    self.sess.put.assert_called_with(url, json=body, microversion=self.sess.default_microversion)