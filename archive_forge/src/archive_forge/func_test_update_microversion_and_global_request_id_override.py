import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_update_microversion_and_global_request_id_override(self):
    patch = {'op': 'replace', 'value': NEW_ATTRIBUTE_VALUE, 'path': '/attribute1'}
    resource = self.manager.update(testable_resource_id=TESTABLE_RESOURCE['uuid'], patch=patch, os_ironic_api_version='1.9', global_request_id=REQ_ID)
    expect = [('PATCH', '/v1/testableresources/%s' % TESTABLE_RESOURCE['uuid'], {'X-OpenStack-Ironic-API-Version': '1.9', 'X-Openstack-Request-Id': REQ_ID}, patch)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NEW_ATTRIBUTE_VALUE, resource.attribute1)