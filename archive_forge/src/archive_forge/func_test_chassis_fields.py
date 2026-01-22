from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_chassis_fields(self):
    self.create_chassis(description='something')
    result = self.conn.baremetal.chassis(fields=['uuid', 'extra'])
    for ch in result:
        self.assertIsNotNone(ch.id)
        self.assertIsNone(ch.description)