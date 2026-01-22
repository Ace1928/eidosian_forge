import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def create_nic(self, vm_name, nic_name, mac_address=None):
    """Create a (synthetic) nic and attach it to the vm.

        :param vm_name: The VM name to which the NIC will be attached to.
        :param nic_name: The name of the NIC to be attached.
        :param mac_address: The VM NIC's MAC address. If None, a Dynamic MAC
            address will be used instead.
        """
    new_nic_data = self._get_new_setting_data(self._SYNTHETIC_ETHERNET_PORT_SETTING_DATA_CLASS)
    new_nic_data.ElementName = nic_name
    new_nic_data.VirtualSystemIdentifiers = ['{' + str(uuid.uuid4()) + '}']
    if mac_address:
        new_nic_data.Address = mac_address.replace(':', '')
        new_nic_data.StaticMacAddress = 'True'
    vmsettings = self._lookup_vm_check(vm_name)
    self._jobutils.add_virt_resource(new_nic_data, vmsettings)