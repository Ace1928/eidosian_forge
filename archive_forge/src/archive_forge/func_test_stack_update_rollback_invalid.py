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
def test_stack_update_rollback_invalid(self):
    arglist = ['my_stack', '-t', self.template_path, '--rollback', 'foo']
    kwargs = copy.deepcopy(self.defaults)
    kwargs['disable_rollback'] = False
    parsed_args = self.check_parser(self.cmd, arglist, [])
    ex = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
    self.assertEqual('--rollback invalid value: foo', str(ex))