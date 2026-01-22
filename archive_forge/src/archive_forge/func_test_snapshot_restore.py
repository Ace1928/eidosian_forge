from unittest import mock
import io
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import snapshot
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
def test_snapshot_restore(self):
    arglist = ['my_stack', 'my_snapshot']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.stack_client.restore.assert_called_with(snapshot_id='my_snapshot', stack_id='my_stack')