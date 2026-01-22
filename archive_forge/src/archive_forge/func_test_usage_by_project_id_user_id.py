import operator
import uuid
from osc_placement.tests.functional import base
def test_usage_by_project_id_user_id(self):
    c1 = str(uuid.uuid4())
    c2 = str(uuid.uuid4())
    c3 = str(uuid.uuid4())
    p1 = str(uuid.uuid4())
    p2 = str(uuid.uuid4())
    u1 = str(uuid.uuid4())
    u2 = str(uuid.uuid4())
    rp = self.resource_provider_create()
    self.resource_inventory_set(rp['uuid'], 'VCPU=16')
    self.resource_allocation_set(c1, ['rp={},VCPU=2'.format(rp['uuid'])], project_id=p1, user_id=u1)
    self.resource_allocation_set(c2, ['rp={},VCPU=4'.format(rp['uuid'])], project_id=p2, user_id=u1)
    self.resource_allocation_set(c3, ['rp={},VCPU=6'.format(rp['uuid'])], project_id=p1, user_id=u2)
    self.assertEqual(12, self.resource_provider_show_usage(uuid=rp['uuid'])[0]['usage'])
    self.assertEqual(8, self.resource_show_usage(project_id=p1)[0]['usage'])
    self.assertEqual(2, self.resource_show_usage(project_id=p1, user_id=u1)[0]['usage'])
    self.assertEqual(4, self.resource_show_usage(project_id=p2)[0]['usage'])