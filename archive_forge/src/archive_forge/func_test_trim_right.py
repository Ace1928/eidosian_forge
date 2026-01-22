from yaql.language import exceptions
import yaql.tests
def test_trim_right(self):
    self.assertEqual('  x', self.eval("'  x  '.trimRight()"))
    self.assertEqual('abx', self.eval("'abxba'.trimRight(ab)"))