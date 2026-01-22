from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
def test_exercise_base_lifecycle_plugin_class(self):
    lcp = lifecycle_plugin.LifecyclePlugin()
    ordinal = lcp.get_ordinal()
    lcp.do_pre_op(None, None, None)
    lcp.do_post_op(None, None, None)
    self.assertEqual(100, ordinal)