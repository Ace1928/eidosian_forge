import warnings
from os_win import utilsfactory
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator import initiator_connector
from os_brick import utils
def check_valid_device(self, path, *args, **kwargs):
    try:
        with open(path, 'r') as dev:
            dev.read(1)
    except IOError:
        LOG.exception('Failed to access the device on the path %(path)s', {'path': path})
        return False
    return True