import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_namespace_get_all_with_resource_types_filter(self):
    ns_fixture = build_namespace_fixture()
    ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
    self.assertIsNotNone(ns_created, 'Could not create a namespace.')
    self._assert_saved_fields(ns_fixture, ns_created)
    fixture = build_association_fixture()
    created = self.db_api.metadef_resource_type_association_create(self.context, ns_created['namespace'], fixture)
    self.assertIsNotNone(created, 'Could not create an association.')
    rt_filters = {'resource_types': fixture['name']}
    found = self.db_api.metadef_namespace_get_all(self.context, filters=rt_filters, sort_key='created_at')
    self.assertEqual(1, len(found))
    for item in found:
        self._assert_saved_fields(ns_fixture, item)