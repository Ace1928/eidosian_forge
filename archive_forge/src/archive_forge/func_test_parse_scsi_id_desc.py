from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_parse_scsi_id_desc(self):
    vpd_str = '008300240103001060002AC00000000000000EA00000869902140004746573740115000400000001'
    buff = _utils.hex_str_to_byte_array(vpd_str)
    identifiers = self._diskutils._parse_scsi_page_83(buff)
    exp_scsi_id_0 = '60002AC00000000000000EA000008699'
    exp_scsi_id_1 = '74657374'
    exp_scsi_id_2 = '00000001'
    exp_identifiers = [{'protocol': None, 'raw_id_desc_size': 20, 'raw_id': _utils.hex_str_to_byte_array(exp_scsi_id_0), 'code_set': 1, 'type': 3, 'id': exp_scsi_id_0, 'association': 0}, {'protocol': None, 'raw_id_desc_size': 8, 'raw_id': _utils.hex_str_to_byte_array(exp_scsi_id_1), 'code_set': 2, 'type': 4, 'id': 'test', 'association': 1}, {'protocol': None, 'raw_id_desc_size': 8, 'raw_id': _utils.hex_str_to_byte_array(exp_scsi_id_2), 'code_set': 1, 'type': 5, 'id': exp_scsi_id_2, 'association': 1}]
    self.assertEqual(exp_identifiers, identifiers)