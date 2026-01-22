import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_create_region_with_duplicate_id(self):
    new_region = unit.new_region_ref()
    PROVIDERS.catalog_api.create_region(new_region)
    self.assertRaises(exception.Conflict, PROVIDERS.catalog_api.create_region, new_region)