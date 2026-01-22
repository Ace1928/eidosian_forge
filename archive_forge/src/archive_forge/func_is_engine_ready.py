import abc
from oslo_log import log as logging
from oslo_utils import importutils
from os_brick import exception
from os_brick.i18n import _
@abc.abstractmethod
def is_engine_ready(self, **kwargs):
    return