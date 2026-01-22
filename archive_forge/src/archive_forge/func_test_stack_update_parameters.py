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
def test_stack_update_parameters(self):
    template_path = '/'.join(self.template_path.split('/')[:-1]) + '/parameters.yaml'
    arglist = ['my_stack', '-t', template_path, '--parameter', 'p1=a', '--parameter', 'p2=6']
    kwargs = copy.deepcopy(self.defaults)
    kwargs['parameters'] = {'p1': 'a', 'p2': '6'}
    kwargs['template']['parameters'] = {'p1': {'type': 'string'}, 'p2': {'type': 'number'}}
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.stack_client.update.assert_called_with(**kwargs)