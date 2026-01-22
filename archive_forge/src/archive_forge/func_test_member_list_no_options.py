import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_member_list_no_options(self):
    arglist = []
    verifylist = []
    self.assertRaises(osc_test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)