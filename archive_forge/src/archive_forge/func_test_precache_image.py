from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import aggregate
from openstack.tests.unit import base
def test_precache_image(self):
    sot = aggregate.Aggregate(**EXAMPLE)
    sot.precache_images(self.sess, ['1'])
    url = 'os-aggregates/4/images'
    body = {'cache': ['1']}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)