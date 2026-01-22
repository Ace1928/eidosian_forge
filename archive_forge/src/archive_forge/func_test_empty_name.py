import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_empty_name(self):
    context = contexts.Context()
    context[''] = 123
    self.assertEqual(123, context['$'])
    self.assertEqual(123, context[''])
    self.assertEqual(123, context['$1'])