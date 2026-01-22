import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import quota
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_quota_defaults_show(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    rows, data = self.cmd.take_action(parsed_args)
    data = dict(zip(rows, data))
    self.api_mock.quota_defaults_show.assert_called_with()
    self.assertEqual(self.qt_defaults, data)