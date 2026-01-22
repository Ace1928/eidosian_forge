from yaql.language import exceptions
import yaql.tests
def test_set_remove(self):
    self.assertCountEqual([1, 3], self.eval('set(1, 2, 3).remove(2)'))
    self.assertCountEqual([3, None], self.eval('set(1, 2, null, 3).remove(1, 2, 5)'))
    self.assertCountEqual([3], self.eval('set(1, 2, null, 3).remove(1, 2, 5, null)'))
    self.assertCountEqual([1, 3, 2], self.eval('set(1, 2, 3, [1, 2]).remove([1, 2])'))
    self.assertTrue(self.eval('isSet(set(1, 2, 3, [1, 2]).remove([1, 2]))'))