import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_association_create(self):
    ns_fixture = build_namespace_fixture()
    ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
    self.assertIsNotNone(ns_created)
    self._assert_saved_fields(ns_fixture, ns_created)
    assn_fixture = build_association_fixture()
    assn_created = self.db_api.metadef_resource_type_association_create(self.context, ns_created['namespace'], assn_fixture)
    self.assertIsNotNone(assn_created)
    self._assert_saved_fields(assn_fixture, assn_created)