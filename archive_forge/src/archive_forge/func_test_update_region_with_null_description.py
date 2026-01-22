import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_update_region_with_null_description(self):
    """Call ``PATCH /regions/{region_id}``."""
    region = unit.new_region_ref(description=None)
    del region['id']
    r = self.patch('/regions/%(region_id)s' % {'region_id': self.region_id}, body={'region': region})
    region['description'] = ''
    self.assertValidRegionResponse(r, region)