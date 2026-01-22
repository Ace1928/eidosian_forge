from yaql.language import exceptions
import yaql.tests
def test_distinct_with_selector(self):
    data = [['a', 1], ['b', 2], ['c', 1], ['d', 3], ['e', 2]]
    self.assertCountEqual([['a', 1], ['b', 2], ['d', 3]], self.eval('$.distinct($[1])', data=data))
    self.assertCountEqual([['a', 1], ['b', 2], ['d', 3]], self.eval('distinct($, $[1])', data=data))