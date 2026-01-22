import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_multi_context_keys(self):
    mc = self.create_multi_context()
    self.assertCountEqual(['$key4', '$key'], mc.keys())
    self.assertCountEqual(['$key2'], mc.parent.keys())
    self.assertCountEqual(['$key3'], mc.parent.parent.keys())