import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
def test__get_as_dict_microversion_and_global_request_id_override(self):
    resource_id = TESTABLE_RESOURCE['uuid']
    resource = self.manager._get_as_dict(resource_id, os_ironic_api_version='1.21', global_request_id=REQ_ID)
    expect = [('GET', '/v1/testableresources/%s' % resource_id, {'X-OpenStack-Ironic-API-Version': '1.21', 'X-Openstack-Request-Id': REQ_ID}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(TESTABLE_RESOURCE, resource)