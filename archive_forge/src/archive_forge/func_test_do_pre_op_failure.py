from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
def test_do_pre_op_failure(self):
    lcp_mappings = [('A::B::C5', TestLifecycleCallout1), ('A::B::C4', TestLifecycleCallout4)]
    self.mock_lcp_class_map(lcp_mappings)
    mc = mock.Mock()
    mc.__setattr__('pre_counter_for_unit_test', 0)
    mc.__setattr__('post_counter_for_unit_test', 0)
    ms = mock.Mock()
    ms.__setattr__('action', 'A')
    failed = False
    try:
        lifecycle_plugin_utils.do_pre_ops(mc, ms, None, None)
    except Exception:
        failed = True
    self.assertTrue(failed)
    self.assertEqual(1, mc.pre_counter_for_unit_test)
    self.assertEqual(1, mc.post_counter_for_unit_test)
    self.mock_get_plugins.assert_called_once_with()