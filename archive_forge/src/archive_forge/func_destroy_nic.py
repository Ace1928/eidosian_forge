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
def destroy_nic(self, vm_name, nic_name):
    """Destroys the NIC with the given nic_name from the given VM.

        :param vm_name: The name of the VM which has the NIC to be destroyed.
        :param nic_name: The NIC's ElementName.
        """
    try:
        nic_data = self._get_nic_data_by_name(nic_name)
        self._jobutils.remove_virt_resource(nic_data)
    except exceptions.NotFound:
        LOG.debug("Ignoring NotFound exception while attempting to remove vm nic: '%s'. It may have been already deleted.", nic_name)