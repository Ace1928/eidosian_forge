from yaql.language import exceptions
import yaql.tests
def test_insert_iter(self):
    self.assertEqual([1, 'a', 2], self.eval('[1, 2].select($).insert(1, a)'))
    self.assertEqual([1, ['a', 'b'], 2], self.eval('[1, 2].select($).insert(1, [a, b])'))
    self.assertEqual([1, 2], self.eval('[1, 2].select($).insert(-1, a)'))
    self.assertEqual([1, 2, 'a'], self.eval('[1, 2].select($).insert(100, a)'))
    self.assertEqual(['b', 'a'], self.eval('[].select($).insert(0, a).insert(0, b)'))
    self.assertEqual(['a', 'a', 'b'], self.eval('set(a, b).orderBy($).insert(1, a)'))