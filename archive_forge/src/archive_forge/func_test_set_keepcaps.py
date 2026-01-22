from unittest import mock
from oslotest import base
from oslo_privsep import capabilities
@mock.patch('oslo_privsep.capabilities._prctl')
def test_set_keepcaps(self, mock_prctl):
    mock_prctl.return_value = 0
    capabilities.set_keepcaps(True)
    self.assertEqual(1, mock_prctl.call_count)
    self.assertCountEqual([8, 1], [int(x) for x in mock_prctl.call_args[0]])