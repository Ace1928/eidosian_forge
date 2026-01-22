import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_network_trunk_subports_without_required_key_fail(self):
    subport = self._trunk['sub_ports'][0]
    arglist = ['--subport', 'segmentation-type=%(seg_type)s,segmentation-id=%(seg_id)s' % {'seg_id': subport['segmentation_id'], 'seg_type': subport['segmentation_type']}, self._trunk['name']]
    verifylist = [('trunk', self._trunk['name']), ('set_subports', [{'segmentation-id': str(subport['segmentation_id']), 'segmentation-type': subport['segmentation_type']}])]
    with testtools.ExpectedException(test_utils.ParserException):
        self.check_parser(self.cmd, arglist, verifylist)
    self.network_client.add_trunk_subports.assert_not_called()