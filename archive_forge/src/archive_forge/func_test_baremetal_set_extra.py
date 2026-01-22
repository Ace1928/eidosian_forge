import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_set_extra(self):
    arglist = ['port', '--extra', 'foo=bar']
    verifylist = [('port', 'port'), ('extra', ['foo=bar'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.port.update.assert_called_once_with('port', [{'path': '/extra/foo', 'value': 'bar', 'op': 'add'}])