import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
@mock.patch.object(commonutils, 'get_from_stdin', autospec=True)
@mock.patch.object(commonutils, 'handle_json_or_file_arg', autospec=True)
def test_baremetal_set_target_raid_config_stdin_exception(self, mock_handle, mock_stdin):
    self.cmd.log = mock.Mock(autospec=True)
    target_value = '-'
    mock_stdin.side_effect = exc.InvalidAttribute('bad')
    arglist = ['node_uuid', '--target-raid-config', target_value]
    verifylist = [('nodes', ['node_uuid']), ('target_raid_config', target_value)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exc.InvalidAttribute, self.cmd.take_action, parsed_args)
    self.cmd.log.warning.assert_not_called()
    mock_stdin.assert_called_once_with('target_raid_config')
    self.assertFalse(mock_handle.called)
    self.assertFalse(self.baremetal_mock.node.set_target_raid_config.called)
    self.assertFalse(self.baremetal_mock.node.update.called)