import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_set_no_options(self):
    arglist = []
    verifylist = []
    self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)