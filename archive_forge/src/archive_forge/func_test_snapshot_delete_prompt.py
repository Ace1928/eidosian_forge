from unittest import mock
import io
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import snapshot
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
@mock.patch('sys.stdin', spec=io.StringIO)
def test_snapshot_delete_prompt(self, mock_stdin):
    arglist = ['my_stack', 'snapshot_id']
    mock_stdin.isatty.return_value = True
    mock_stdin.readline.return_value = 'y'
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    mock_stdin.readline.assert_called_with()
    self.stack_client.snapshot_delete.assert_called_with('my_stack', 'snapshot_id')