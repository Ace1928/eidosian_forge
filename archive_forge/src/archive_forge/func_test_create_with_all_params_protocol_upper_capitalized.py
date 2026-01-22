import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallrule
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_create_with_all_params_protocol_upper_capitalized(self):
    for protocol in ('TCP', 'Tcp', 'ANY', 'AnY', 'iCMp'):
        arglist, verifylist = self._set_all_params({'protocol': protocol})
        self.assertRaises(testtools.matchers._impl.MismatchError, self.check_parser, self.cmd, arglist, verifylist)