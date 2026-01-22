import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_regions_with_same_description_string(self):
    """Call ``POST /regions`` with duplicate descriptions."""
    region_desc = 'Some Region Description'
    ref1 = unit.new_region_ref(description=region_desc)
    ref2 = unit.new_region_ref(description=region_desc)
    resp1 = self.post('/regions', body={'region': ref1})
    self.assertValidRegionResponse(resp1, ref1)
    resp2 = self.post('/regions', body={'region': ref2})
    self.assertValidRegionResponse(resp2, ref2)