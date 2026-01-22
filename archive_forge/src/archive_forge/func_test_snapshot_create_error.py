from unittest import mock
import io
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import snapshot
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
def test_snapshot_create_error(self):
    arglist = ['my_stack', '--name', 'test_snapshot']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.stack_client.snapshot.side_effect = heat_exc.HTTPNotFound
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)