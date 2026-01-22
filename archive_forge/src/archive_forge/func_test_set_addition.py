from yaql.language import exceptions
import yaql.tests
def test_set_addition(self):
    self.assertCountEqual([4, 3, 2, 1], self.eval('set(1, 2, 3) + set(4, 2, 3)'))
    self.assertTrue(self.eval('isSet(set(1, 2, 3) + set(4, 2, 3))'))