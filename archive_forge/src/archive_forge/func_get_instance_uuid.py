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
def get_instance_uuid(self, vm_name):
    instance_notes = self._get_instance_notes(vm_name)
    if instance_notes and uuidutils.is_uuid_like(instance_notes[0]):
        return instance_notes[0]