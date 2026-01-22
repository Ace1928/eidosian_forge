import warnings
from os_win import utilsfactory
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator import initiator_connector
from os_brick import utils
@staticmethod
def check_multipath_support(enforce_multipath):
    hostutils = utilsfactory.get_hostutils()
    mpio_enabled = hostutils.check_server_feature(hostutils.FEATURE_MPIO)
    if not mpio_enabled:
        err_msg = _('Using multipath connections for iSCSI and FC disks requires the Multipath IO Windows feature to be enabled. MPIO must be configured to claim such devices.')
        LOG.error(err_msg)
        if enforce_multipath:
            raise exception.BrickException(err_msg)
        return False
    return True