import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_child_contexts(self):
    context = contexts.Context()
    context2 = context.create_child_context()
    context['key'] = 123
    self.assertEqual(123, context2['key'])
    context2['key'] = 345
    self.assertEqual(345, context2['key'])
    del context2['key']
    self.assertEqual(123, context2['key'])