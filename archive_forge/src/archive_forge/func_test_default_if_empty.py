from yaql.language import exceptions
import yaql.tests
def test_default_if_empty(self):
    self.assertEqual([1, 2], self.eval('[].defaultIfEmpty([1, 2])'))
    self.assertEqual([3, 4], self.eval('[3, 4].defaultIfEmpty([1, 2])'))
    self.assertEqual([1, 2], self.eval('[].select($).defaultIfEmpty([1, 2])'))
    self.assertEqual([3, 4], self.eval('[3, 4].select($).defaultIfEmpty([1, 2])'))