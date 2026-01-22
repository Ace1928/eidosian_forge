import testtools
from barbicanclient import formatter
def test_should_get_list_objects(self):
    entity_1 = Entity('test_attr_a_1', 'test_attr_b_1', 'test_attr_c_1')
    entity_2 = Entity('test_attr_a_2', 'test_attr_b_2', 'test_attr_c_2')
    columns, data = EntityFormatter._list_objects([entity_1, entity_2])
    self.assertEqual(('Column A', 'Column B', 'Column C'), columns)
    self.assertEqual([('test_attr_a_1', 'test_attr_b_1', 'test_attr_c_1'), ('test_attr_a_2', 'test_attr_b_2', 'test_attr_c_2')], [e for e in data])