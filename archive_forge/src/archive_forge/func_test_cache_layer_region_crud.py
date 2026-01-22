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
def test_cache_layer_region_crud(self):
    new_region = unit.new_region_ref()
    region_id = new_region['id']
    PROVIDERS.catalog_api.create_region(new_region.copy())
    updated_region = copy.deepcopy(new_region)
    updated_region['description'] = uuid.uuid4().hex
    PROVIDERS.catalog_api.get_region(region_id)
    PROVIDERS.catalog_api.driver.update_region(region_id, updated_region)
    self.assertLessEqual(new_region.items(), PROVIDERS.catalog_api.get_region(region_id).items())
    PROVIDERS.catalog_api.get_region.invalidate(PROVIDERS.catalog_api, region_id)
    self.assertLessEqual(updated_region.items(), PROVIDERS.catalog_api.get_region(region_id).items())
    PROVIDERS.catalog_api.driver.delete_region(region_id)
    self.assertLessEqual(updated_region.items(), PROVIDERS.catalog_api.get_region(region_id).items())
    PROVIDERS.catalog_api.get_region.invalidate(PROVIDERS.catalog_api, region_id)
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_id)