import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_set_no_options(self):
    arglist = []
    verifylist = []
    self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)