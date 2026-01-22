import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_create_no_options(self):
    arglist = []
    verifylist = []
    self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)