from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.block_storage.v3 import group_type as _group_type
from openstack.block_storage.v3 import volume as _volume
from openstack.tests.functional.block_storage.v3 import base
def test_group_type(self):
    group_type = self.conn.block_storage.get_group_type(self.group_type.id)
    self.assertEqual(self.group_type.name, group_type.name)
    group_type = self.conn.block_storage.find_group_type(self.group_type.name)
    self.assertEqual(self.group_type.id, group_type.id)
    group_types = list(self.conn.block_storage.group_types())
    self.assertIn(self.group_type.id, {g.id for g in group_types})
    group_type_name = self.getUniqueString()
    group_type_description = self.getUniqueString()
    group_type = self.conn.block_storage.update_group_type(self.group_type, name=group_type_name, description=group_type_description)
    self.assertIsInstance(group_type, _group_type.GroupType)
    group_type = self.conn.block_storage.get_group_type(self.group_type.id)
    self.assertEqual(group_type_name, group_type.name)
    self.assertEqual(group_type_description, group_type.description)