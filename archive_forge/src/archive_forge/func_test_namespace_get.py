import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_namespace_get(self):
    fixture = build_namespace_fixture()
    created = self.db_api.metadef_namespace_create(self.context, fixture)
    self.assertIsNotNone(created)
    self._assert_saved_fields(fixture, created)
    found = self.db_api.metadef_namespace_get(self.context, created['namespace'])
    self.assertIsNotNone(found, 'Namespace not found.')