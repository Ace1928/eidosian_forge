import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_delete_microversion_and_global_request_id_override(self):
    resource = self.manager.delete(testable_resource_id=TESTABLE_RESOURCE['uuid'], os_ironic_api_version='1.9', global_request_id=REQ_ID)
    expect = [('DELETE', '/v1/testableresources/%s' % TESTABLE_RESOURCE['uuid'], {'X-OpenStack-Ironic-API-Version': '1.9', 'X-Openstack-Request-Id': REQ_ID}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(resource)