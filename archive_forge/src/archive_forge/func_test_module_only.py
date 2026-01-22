from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_module_only(self):
    import sys
    self.assertIs(sys, get_named_object('sys'))