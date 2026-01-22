import copy
import io
from unittest import mock
from osc_lib import exceptions as exc
from osc_lib import utils
import testscenarios
import yaml
from heatclient.common import template_format
from heatclient import exc as heat_exc
from heatclient.osc.v1 import stack
from heatclient.tests import inline_templates
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import events
from heatclient.v1 import resources
from heatclient.v1 import stacks
@mock.patch('sys.stdin', spec=io.StringIO)
def test_stack_delete_prompt_no(self, mock_stdin):
    arglist = ['my_stack']
    mock_stdin.isatty.return_value = True
    mock_stdin.readline.return_value = 'n'
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    mock_stdin.readline.assert_called_with()
    self.stack_client.delete.assert_not_called()