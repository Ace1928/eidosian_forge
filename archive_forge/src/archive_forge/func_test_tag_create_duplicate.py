import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_tag_create_duplicate(self):
    fixture = build_namespace_fixture()
    created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
    self.assertIsNotNone(created_ns)
    self._assert_saved_fields(fixture, created_ns)
    fixture_tag = build_tag_fixture(namespace_id=created_ns['id'])
    created_tag = self.db_api.metadef_tag_create(self.context, created_ns['namespace'], fixture_tag)
    self._assert_saved_fields(fixture_tag, created_tag)
    self.assertRaises(exception.Duplicate, self.db_api.metadef_tag_create, self.context, created_ns['namespace'], fixture_tag)