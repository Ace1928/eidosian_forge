import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_deploy_template_show(self):
    arglist = [baremetal_fakes.baremetal_deploy_template_uuid]
    verifylist = [('template', baremetal_fakes.baremetal_deploy_template_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = [baremetal_fakes.baremetal_deploy_template_uuid]
    self.baremetal_mock.deploy_template.get.assert_called_with(*args, fields=None)
    collist = ('extra', 'name', 'steps', 'uuid')
    self.assertEqual(collist, columns)
    datalist = (baremetal_fakes.baremetal_deploy_template_extra, baremetal_fakes.baremetal_deploy_template_name, baremetal_fakes.baremetal_deploy_template_steps, baremetal_fakes.baremetal_deploy_template_uuid)
    self.assertEqual(datalist, tuple(data))