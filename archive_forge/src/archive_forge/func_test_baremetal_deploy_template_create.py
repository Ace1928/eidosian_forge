import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_deploy_template_create(self):
    arglist = [baremetal_fakes.baremetal_deploy_template_name, '--steps', baremetal_fakes.baremetal_deploy_template_steps]
    verifylist = [('name', baremetal_fakes.baremetal_deploy_template_name), ('steps', baremetal_fakes.baremetal_deploy_template_steps)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'name': baremetal_fakes.baremetal_deploy_template_name, 'steps': json.loads(baremetal_fakes.baremetal_deploy_template_steps)}
    self.baremetal_mock.deploy_template.create.assert_called_once_with(**args)