import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
@unit.skip_if_cache_disabled('catalog')
def test_invalidate_cache_when_updating_region(self):
    new_region = unit.new_region_ref()
    region_id = new_region['id']
    PROVIDERS.catalog_api.create_region(new_region)
    PROVIDERS.catalog_api.get_region(region_id)
    new_description = {'description': uuid.uuid4().hex}
    PROVIDERS.catalog_api.update_region(region_id, new_description)
    current_region = PROVIDERS.catalog_api.get_region(region_id)
    self.assertEqual(new_description['description'], current_region['description'])