import testtools
from barbicanclient import formatter
def test_should_get_str(self):
    entity = Entity('test_attr_a_1', 'test_attr_b_1', 'test_attr_c_1')
    self.assertEqual('+----------+---------------+\n| Field    | Value         |\n+----------+---------------+\n| Column A | test_attr_a_1 |\n| Column B | test_attr_b_1 |\n| Column C | test_attr_c_1 |\n+----------+---------------+', str(entity))