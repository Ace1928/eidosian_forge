import copy
import json
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_deploy_template
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_deploy_template_list_fields_multiple(self):
    arglist = ['--fields', 'uuid', 'name', '--fields', 'steps']
    verifylist = [('fields', [['uuid', 'name'], ['steps']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'name', 'steps')}
    self.baremetal_mock.deploy_template.list.assert_called_with(**kwargs)