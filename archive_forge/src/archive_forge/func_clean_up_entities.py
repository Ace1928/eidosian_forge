import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def clean_up_entities(self):
    """Clean up entity test data from Limit Test Cases."""
    for entity in self.ENTITIES:
        self._delete_test_data(entity, self.entity_lists[entity])
    del self.entity_lists