import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_namespace_delete_with_content(self):
    fixture_ns = build_namespace_fixture()
    created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
    self._assert_saved_fields(fixture_ns, created_ns)
    fixture_obj = build_object_fixture()
    created_obj = self.db_api.metadef_object_create(self.context, created_ns['namespace'], fixture_obj)
    self.assertIsNotNone(created_obj)
    fixture_prop = build_property_fixture(namespace_id=created_ns['id'])
    created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], fixture_prop)
    self.assertIsNotNone(created_prop)
    fixture_assn = build_association_fixture()
    created_assn = self.db_api.metadef_resource_type_association_create(self.context, created_ns['namespace'], fixture_assn)
    self.assertIsNotNone(created_assn)
    deleted_ns = self.db_api.metadef_namespace_delete(self.context, created_ns['namespace'])
    self.assertRaises(exception.NotFound, self.db_api.metadef_namespace_get, self.context, deleted_ns['namespace'])