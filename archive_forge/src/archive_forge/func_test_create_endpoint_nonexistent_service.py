import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_create_endpoint_nonexistent_service(self):
    endpoint = unit.new_endpoint_ref(service_id=uuid.uuid4().hex, region_id=None)
    self.assertRaises(exception.ValidationError, PROVIDERS.catalog_api.create_endpoint, endpoint['id'], endpoint)