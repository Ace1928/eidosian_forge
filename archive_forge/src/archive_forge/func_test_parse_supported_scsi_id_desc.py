from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_parse_supported_scsi_id_desc(self):
    vpd_str = '008300240103001060002AC00000000000000EA00000869901140004000003F40115000400000001'
    buff = _utils.hex_str_to_byte_array(vpd_str)
    identifiers = self._diskutils._parse_scsi_page_83(buff, select_supported_identifiers=True)
    exp_scsi_id = '60002AC00000000000000EA000008699'
    exp_identifiers = [{'protocol': None, 'raw_id_desc_size': 20, 'raw_id': _utils.hex_str_to_byte_array(exp_scsi_id), 'code_set': 1, 'type': 3, 'id': exp_scsi_id, 'association': 0}]
    self.assertEqual(exp_identifiers, identifiers)