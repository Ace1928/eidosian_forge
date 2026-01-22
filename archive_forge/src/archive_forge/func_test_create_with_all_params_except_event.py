import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
def test_create_with_all_params_except_event(self):
    arglist, verifylist = self._set_all_params({'event': ''})
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)