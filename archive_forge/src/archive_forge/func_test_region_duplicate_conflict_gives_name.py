import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_region_duplicate_conflict_gives_name(self):
    region_ref = unit.new_region_ref()
    PROVIDERS.catalog_api.create_region(region_ref)
    try:
        PROVIDERS.catalog_api.create_region(region_ref)
    except exception.Conflict as e:
        self.assertIn('Duplicate ID, %s' % region_ref['id'], repr(e))
    else:
        self.fail('Create duplicate region did not raise a conflict')