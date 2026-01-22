import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import amphora
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_amphora_attrs')
def test_amphora_list_with_loadbalancer(self, mock_client):
    mock_client.return_value = {'loadbalancer_id': self._amp.loadbalancer_id, 'compute_id': self._amp.compute_id, 'role': self._amp.role, 'status': self._amp.status}
    arglist = ['--loadbalancer', self._amp.loadbalancer_id, '--compute-id', self._amp.compute_id, '--role', 'Master', '--status', 'allocAted']
    verify_list = [('loadbalancer', self._amp.loadbalancer_id), ('compute_id', self._amp.compute_id), ('role', 'MASTER'), ('status', 'ALLOCATED')]
    parsed_args = self.check_parser(self.cmd, arglist, verify_list)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data_list, tuple(data))