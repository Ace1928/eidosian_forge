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
def create_scsi_controller(self, vm_name):
    """Create an iscsi controller ready to mount volumes."""
    vmsettings = self._lookup_vm_check(vm_name)
    scsicontrl = self._get_new_resource_setting_data(self._SCSI_CTRL_RES_SUB_TYPE)
    scsicontrl.VirtualSystemIdentifiers = ['{' + str(uuid.uuid4()) + '}']
    self._jobutils.add_virt_resource(scsicontrl, vmsettings)