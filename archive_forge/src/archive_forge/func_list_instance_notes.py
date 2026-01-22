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
def list_instance_notes(self):
    instance_notes = []
    for vs in self._conn.Msvm_VirtualSystemSettingData(['ElementName', 'Notes'], VirtualSystemType=self._VIRTUAL_SYSTEM_TYPE_REALIZED):
        vs_notes = vs.Notes
        vs_name = vs.ElementName
        if vs_notes is not None and vs_name:
            instance_notes.append((vs_name, [v for v in vs_notes if v]))
    return instance_notes