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
def test_stack_adopt_defaults(self):
    arglist = ['my_stack', '--adopt-file', self.adopt_file]
    cols = ['id', 'stack_name', 'description', 'creation_time', 'updated_time', 'stack_status', 'stack_status_reason']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.stack_client.create.assert_called_with(**self.defaults)
    self.assertEqual(cols, columns)