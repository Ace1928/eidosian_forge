from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
def test_get_plug_point_class_instances(self):
    """Tests the get_plug_point_class_instances function."""
    lcp_mappings = [('A::B::C1', TestLifecycleCallout1)]
    self.mock_lcp_class_map(lcp_mappings)
    pp_cinstances = lifecycle_plugin_utils.get_plug_point_class_instances()
    self.assertIsNotNone(pp_cinstances)
    self.assertTrue(self.is_iterable(pp_cinstances), 'not iterable: %s' % pp_cinstances)
    self.assertEqual(1, len(pp_cinstances))
    self.assertEqual(TestLifecycleCallout1, pp_cinstances[0].__class__)
    self.mock_get_plugins.assert_called_once_with()