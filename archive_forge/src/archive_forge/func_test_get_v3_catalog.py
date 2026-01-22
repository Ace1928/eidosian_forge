from unittest import mock
import uuid
from keystone.catalog.backends import base as catalog_base
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_get_v3_catalog(self):
    user_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    catalog_ref = PROVIDERS.catalog_api.get_v3_catalog(user_id, project_id)
    exp_catalog = [{'endpoints': [{'interface': 'admin', 'region': 'RegionOne', 'url': 'http://localhost:8774/v1.1/%s' % project_id}, {'interface': 'public', 'region': 'RegionOne', 'url': 'http://localhost:8774/v1.1/%s' % project_id}, {'interface': 'internal', 'region': 'RegionOne', 'url': 'http://localhost:8774/v1.1/%s' % project_id}], 'type': 'compute', 'name': "'Compute Service'", 'id': '2'}, {'endpoints': [{'interface': 'admin', 'region': 'RegionOne', 'url': 'http://localhost:35357/v3'}, {'interface': 'public', 'region': 'RegionOne', 'url': 'http://localhost:5000/v3'}, {'interface': 'internal', 'region': 'RegionOne', 'url': 'http://localhost:35357/v3'}], 'type': 'identity', 'name': "'Identity Service'", 'id': '1'}]
    self.assert_catalogs_equal(exp_catalog, catalog_ref)