import platform
import socket
import sys
from oslo_log import log as logging
from oslo_utils import importutils
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick import utils
def get_connector_mapping(arch=None):
    """Get connector mapping based on platform.

    This is used by Nova to get the right connector information.

    :param arch: The architecture being requested.
    """
    if arch is None:
        arch = platform.machine()
    if sys.platform == 'win32':
        return _connector_mapping_windows
    elif arch in (initiator.S390, initiator.S390X):
        return _connector_mapping_linux_s390x
    elif arch in (initiator.PPC64, initiator.PPC64LE):
        return _connector_mapping_linux_ppc64
    else:
        return _connector_mapping_linux