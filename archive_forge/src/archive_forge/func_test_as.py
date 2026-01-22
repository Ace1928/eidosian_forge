from yaql.language import exceptions
import yaql.tests
def test_as(self):
    self.assertEqual([3, 6], self.eval('[1, 2].as(sum($) => a).select($ * $a)'))