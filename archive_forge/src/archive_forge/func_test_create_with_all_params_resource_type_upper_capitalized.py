import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
def test_create_with_all_params_resource_type_upper_capitalized(self):
    for res_type in ('SECURITY_GROUP', 'Security_group', 'security_Group'):
        arglist, verifylist = self._set_all_params({'resource_type': res_type})
        self.assertRaises(testtools.matchers._impl.MismatchError, self.check_parser, self.cmd, arglist, verifylist)