from yaql.language import exceptions
import yaql.tests
def test_set_from_iterator(self):
    self.assertCountEqual([2, 1, 3], self.eval('set([1, 2, 3].select($))'))