from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_create_default_setting_data')
def test_prepare_profile_sd(self, mock_create_default_sd):
    mock_profile_sd = mock_create_default_sd.return_value
    actual_profile_sd = self.netutils._prepare_profile_sd(profile_id=mock.sentinel.profile_id, profile_data=mock.sentinel.profile_data, profile_name=mock.sentinel.profile_name, net_cfg_instance_id=mock.sentinel.net_cfg_instance_id, cdn_label_id=mock.sentinel.cdn_label_id, cdn_label_string=mock.sentinel.cdn_label_string, vendor_id=mock.sentinel.vendor_id, vendor_name=mock.sentinel.vendor_name)
    self.assertEqual(mock_profile_sd, actual_profile_sd)
    self.assertEqual(mock.sentinel.profile_id, mock_profile_sd.ProfileId)
    self.assertEqual(mock.sentinel.profile_data, mock_profile_sd.ProfileData)
    self.assertEqual(mock.sentinel.profile_name, mock_profile_sd.ProfileName)
    self.assertEqual(mock.sentinel.net_cfg_instance_id, mock_profile_sd.NetCfgInstanceId)
    self.assertEqual(mock.sentinel.cdn_label_id, mock_profile_sd.CdnLabelId)
    self.assertEqual(mock.sentinel.cdn_label_string, mock_profile_sd.CdnLabelString)
    self.assertEqual(mock.sentinel.vendor_id, mock_profile_sd.VendorId)
    self.assertEqual(mock.sentinel.vendor_name, mock_profile_sd.VendorName)
    mock_create_default_sd.assert_called_once_with(self.netutils._PORT_PROFILE_SET_DATA)