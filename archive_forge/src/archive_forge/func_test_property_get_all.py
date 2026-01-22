import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_property_get_all(self):
    ns_fixture = build_namespace_fixture()
    ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
    self.assertIsNotNone(ns_created, 'Could not create a namespace.')
    self._assert_saved_fields(ns_fixture, ns_created)
    fixture1 = build_property_fixture(namespace_id=ns_created['id'])
    created_p1 = self.db_api.metadef_property_create(self.context, ns_created['namespace'], fixture1)
    self.assertIsNotNone(created_p1, 'Could not create a property.')
    fixture2 = build_property_fixture(namespace_id=ns_created['id'], name='test-prop-2')
    created_p2 = self.db_api.metadef_property_create(self.context, ns_created['namespace'], fixture2)
    self.assertIsNotNone(created_p2, 'Could not create a property.')
    found = self.db_api.metadef_property_get_all(self.context, ns_created['namespace'])
    self.assertEqual(2, len(found))