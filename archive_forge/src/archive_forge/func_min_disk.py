from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
@min_disk.setter
def min_disk(self, value):
    if value and value < 0:
        extra_msg = _('Cannot be a negative value')
        raise exception.InvalidParameterValue(value=value, param='min_disk', extra_msg=extra_msg)
    self._min_disk = value