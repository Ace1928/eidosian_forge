import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_tag_get_all(self):
    ns_fixture = build_namespace_fixture()
    ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
    self.assertIsNotNone(ns_created, 'Could not create a namespace.')
    self._assert_saved_fields(ns_fixture, ns_created)
    fixture1 = build_tag_fixture(namespace_id=ns_created['id'])
    created_tag1 = self.db_api.metadef_tag_create(self.context, ns_created['namespace'], fixture1)
    self.assertIsNotNone(created_tag1, 'Could not create tag 1.')
    fixture2 = build_tag_fixture(namespace_id=ns_created['id'], name='test-tag-2')
    created_tag2 = self.db_api.metadef_tag_create(self.context, ns_created['namespace'], fixture2)
    self.assertIsNotNone(created_tag2, 'Could not create tag 2.')
    found = self.db_api.metadef_tag_get_all(self.context, ns_created['namespace'], sort_key='created_at')
    self.assertEqual(2, len(found))