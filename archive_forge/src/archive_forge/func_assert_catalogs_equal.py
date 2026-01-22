from unittest import mock
import uuid
from keystone.catalog.backends import base as catalog_base
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def assert_catalogs_equal(self, expected, observed):

    def sort_key(d):
        return d['id']
    for e, o in zip(sorted(expected, key=sort_key), sorted(observed, key=sort_key)):
        expected_endpoints = e.pop('endpoints')
        observed_endpoints = o.pop('endpoints')
        self.assertDictEqual(e, o)
        self.assertCountEqual(expected_endpoints, observed_endpoints)