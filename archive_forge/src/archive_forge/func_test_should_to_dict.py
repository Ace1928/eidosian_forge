import testtools
from barbicanclient import formatter
def test_should_to_dict(self):
    entity = Entity('test_attr_a_1', 'test_attr_b_1', 'test_attr_c_1')
    self.assertEqual({'Column A': 'test_attr_a_1', 'Column B': 'test_attr_b_1', 'Column C': 'test_attr_c_1'}, entity.to_dict())