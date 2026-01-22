import uuid
from osc_placement.tests.functional import base
def test_allocation_update_to_empty(self):
    consumer_uuid = str(uuid.uuid4())
    project_uuid = str(uuid.uuid4())
    user_uuid = str(uuid.uuid4())
    self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(self.rp1['uuid'])], project_id=project_uuid, user_id=user_uuid, consumer_type='INSTANCE')
    result = self.resource_allocation_unset(consumer_uuid, columns=('resources', 'consumer_type'))
    self.assertEqual([], result)