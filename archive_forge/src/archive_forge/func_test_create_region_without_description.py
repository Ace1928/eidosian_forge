import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_region_without_description(self):
    """Call ``POST /regions`` without description in the request body."""
    ref = unit.new_region_ref(description=None)
    del ref['description']
    r = self.post('/regions', body={'region': ref})
    ref['description'] = ''
    self.assertValidRegionResponse(r, ref)