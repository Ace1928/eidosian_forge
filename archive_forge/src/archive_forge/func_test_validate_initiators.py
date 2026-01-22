from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import iscsi
from os_brick.tests.windows import test_base
@ddt.data({'requested_initiators': [mock.sentinel.initiator_0], 'available_initiators': [mock.sentinel.initiator_0, mock.sentinel.initiator_1]}, {'requested_initiators': [mock.sentinel.initiator_0], 'available_initiators': [mock.sentinel.initiator_1]}, {'requested_initiators': [], 'available_initiators': [mock.sentinel.software_initiator]})
@ddt.unpack
def test_validate_initiators(self, requested_initiators, available_initiators):
    self._iscsi_utils.get_iscsi_initiators.return_value = available_initiators
    self._connector.initiator_list = requested_initiators
    expected_valid_initiator = not set(requested_initiators).difference(set(available_initiators))
    valid_initiator = self._connector.validate_initiators()
    self.assertEqual(expected_valid_initiator, valid_initiator)