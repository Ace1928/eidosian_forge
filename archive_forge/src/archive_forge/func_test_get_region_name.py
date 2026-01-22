import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
def test_get_region_name(self):

    def assert_region_name(default, compute):
        self.assertEqual(default, cc.region_name)
        self.assertEqual(default, cc.get_region_name())
        self.assertEqual(default, cc.get_region_name(service_type=None))
        self.assertEqual(compute, cc.get_region_name(service_type='compute'))
        self.assertEqual(default, cc.get_region_name(service_type='placement'))
    cc = cloud_region.CloudRegion(config=fake_services_dict)
    assert_region_name(None, None)
    cc = cloud_region.CloudRegion(region_name='foo', config=fake_services_dict)
    assert_region_name('foo', 'foo')
    services_dict = dict(fake_services_dict, region_name='the-default', compute_region_name='compute-region')
    cc = cloud_region.CloudRegion(config=services_dict)
    assert_region_name('the-default', 'compute-region')
    services_dict = dict(fake_services_dict, region_name='dict', compute_region_name='compute-region')
    cc = cloud_region.CloudRegion(region_name='kwarg', config=services_dict)
    assert_region_name('kwarg', 'compute-region')