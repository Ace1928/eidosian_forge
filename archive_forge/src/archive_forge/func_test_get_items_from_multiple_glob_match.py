import sys
from breezy import rules, tests
def test_get_items_from_multiple_glob_match(self):
    rs = self.make_searcher('[name *.txt *.py \'x x\' "y y"]\nfoo=bar\na=True\n')
    self.assertEqual((), rs.get_items('NEWS'))
    self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('a.py'))
    self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('a.txt'))
    self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('x x'))
    self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('y y'))
    self.assertEqual('bar', rs.get_single_value('a.txt', 'foo'))