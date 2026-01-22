from yaql.language import exceptions
import yaql.tests
def test_group_by_old_syntax(self):
    data = {'a': 1, 'b': 2, 'c': 1, 'd': 3, 'e': 2}
    self.assertCountEqual([[1, 'ac'], [2, 'be'], [3, 'd']], self.eval('$.items().orderBy($[0]).groupBy($[1], $[0], [$[0], $[1].sum()])', data=data))
    self.assertCountEqual([[1, ['a', 1, 'c', 1]], [2, ['b', 2, 'e', 2]], [3, ['d', 3]]], self.eval('$.items().orderBy($[0]).groupBy($[1],,  [$[0], $[1].sum()])', data=data))
    self.assertCountEqual([[1, ['a', 1, 'c', 1]], [2, ['b', 2, 'e', 2]], [3, ['d', 3]]], self.eval('$.items().orderBy($[0]).groupBy($[1], aggregator => [$[0], $[1].sum()])', data=data))