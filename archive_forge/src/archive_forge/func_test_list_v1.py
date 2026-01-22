import re
from tempest.lib import exceptions
from glanceclient.tests.functional import base
def test_list_v1(self):
    out = self.glance('--os-image-api-version 1 image-list')
    endpoints = self.parser.listing(out)
    self.assertTableStruct(endpoints, ['ID', 'Name', 'Disk Format', 'Container Format', 'Size', 'Status'])