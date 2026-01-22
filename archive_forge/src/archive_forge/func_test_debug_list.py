import re
from tempest.lib import exceptions
from glanceclient.tests.functional import base
def test_debug_list(self):
    self.glance('--os-image-api-version 2 image-list', flags='--debug')