import re
from tempest.lib import exceptions
from glanceclient.tests.functional import base
def test_member_list_v1(self):
    tenant_name = '--tenant-id %s' % self.creds['project_name']
    out = self.glance('--os-image-api-version 1 member-list', params=tenant_name)
    endpoints = self.parser.listing(out)
    self.assertTableStruct(endpoints, ['Image ID', 'Member ID', 'Can Share'])