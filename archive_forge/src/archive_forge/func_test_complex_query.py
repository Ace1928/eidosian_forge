from yaql.language import exceptions
import yaql.tests
def test_complex_query(self):
    data = [1, 2, 3, 4, 5, 6]
    self.assertEqual([4], self.eval('$.where($ < 4).select($ * $).skip(1).limit(1)', data=data))