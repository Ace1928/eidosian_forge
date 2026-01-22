import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def remove_all_pci_devices(self, vm_name):
    """Removes all the PCI devices from the given VM.

        :param vm_name: the name of the VM from which all the PCI devices will
            be detached from.
        """
    vmsettings = self._lookup_vm_check(vm_name)
    pci_sds = _wqlutils.get_element_associated_class(self._conn, self._PCI_EXPRESS_SETTING_DATA, vmsettings.InstanceID)
    if pci_sds:
        self._jobutils.remove_multiple_virt_resources(pci_sds)