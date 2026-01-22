import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_key_deletion(self):
    context = contexts.Context()
    context['key'] = 123
    del context['key']
    self.assertIsNone(context['key'])