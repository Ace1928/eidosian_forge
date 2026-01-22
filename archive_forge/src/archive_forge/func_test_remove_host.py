from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import aggregate
from openstack.tests.unit import base
def test_remove_host(self):
    sot = aggregate.Aggregate(**EXAMPLE)
    sot.remove_host(self.sess, 'host1')
    url = 'os-aggregates/4/action'
    body = {'remove_host': {'host': 'host1'}}
    self.sess.post.assert_called_with(url, json=body, microversion=None)