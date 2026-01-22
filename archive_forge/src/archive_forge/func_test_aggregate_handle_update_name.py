from unittest import mock
from heat.engine.clients.os import nova
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_aggregate_handle_update_name(self):
    value = mock.MagicMock()
    self.aggregates.get.return_value = value
    prop_diff = {'name': 'new_host_aggregate', 'metadata': {'availability_zone': 'new_nova'}, 'availability_zone': 'new_nova'}
    expected = {'name': 'new_host_aggregate', 'availability_zone': 'new_nova'}
    self.my_aggregate.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    value.update.assert_called_once_with(expected)