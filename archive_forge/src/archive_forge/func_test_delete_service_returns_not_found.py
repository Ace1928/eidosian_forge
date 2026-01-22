import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_delete_service_returns_not_found(self):
    self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.delete_service, uuid.uuid4().hex)