from unittest import mock
from heat.engine.clients.os import nova
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_aggregate_handle_update_metadata(self):
    ag = mock.MagicMock()
    self.aggregates.get.return_value = ag
    prop_diff = {'metadata': {'availability_zone': 'nova3'}}
    set_metadata_expected = {'availability_zone': 'nova3'}
    self.my_aggregate.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.assertEqual(0, ag.update.call_count)
    self.assertEqual(0, ag.add_host.call_count)
    self.assertEqual(0, ag.remove_host.call_count)
    ag.set_metadata.assert_called_once_with(set_metadata_expected)