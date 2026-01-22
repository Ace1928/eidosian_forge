import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def test_rescan_hosts_port_not_found_driver_disables_wildcards(self):
    """Test when we don't find the target ports but driver forces scan."""
    get_chan_results = [([], {1}), ([], {1})]
    hbas, con_props = self.__get_rescan_info(zone_manager=True)
    con_props['enable_wildcard_scan'] = False
    with mock.patch.object(self.lfc, '_get_hba_channel_scsi_target_lun', side_effect=get_chan_results), mock.patch.object(self.lfc, '_execute', side_effect=None) as execute_mock:
        self.lfc.rescan_hosts(hbas, con_props)
        execute_mock.assert_not_called()