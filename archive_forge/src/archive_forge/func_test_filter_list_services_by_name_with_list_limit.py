import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_filter_list_services_by_name_with_list_limit(self):
    """Call ``GET /services?name=<some name>``."""
    self.config_fixture.config(list_limit=1)
    self.test_filter_list_services_by_name()