import uuid
from keystone.common import driver_hints
from keystone import exception
def test_generates_uuids(self):
    self.assertIs(self.expected_generates_uuids, self.driver.generates_uuids())