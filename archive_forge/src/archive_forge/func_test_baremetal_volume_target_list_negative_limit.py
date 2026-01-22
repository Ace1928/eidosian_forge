import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_list_negative_limit(self):
    arglist = ['--limit', '-1']
    verifylist = [('limit', -1)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)