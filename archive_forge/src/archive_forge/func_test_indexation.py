import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_indexation(self):
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj)
    self.assertEqual('key', self.eval('$[key]', obj))