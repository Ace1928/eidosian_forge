from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_create
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import create_resources
def test_baremetal_create_no_args(self):
    arglist = []
    verifylist = []
    self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)