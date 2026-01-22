import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_deploy_template_set_name(self):
    new_name = 'foo'
    arglist = [baremetal_fakes.baremetal_deploy_template_uuid, '--name', new_name]
    verifylist = [('template', baremetal_fakes.baremetal_deploy_template_uuid), ('name', new_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.deploy_template.update.assert_called_once_with(baremetal_fakes.baremetal_deploy_template_uuid, [{'path': '/name', 'value': new_name, 'op': 'add'}])