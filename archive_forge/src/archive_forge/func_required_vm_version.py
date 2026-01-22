import ctypes
import inspect
from pkg_resources import parse_version
import textwrap
import time
import types
import eventlet
from eventlet import tpool
import netaddr
from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
import six
from os_win import constants
from os_win import exceptions
def required_vm_version(min_version=constants.VM_VERSION_5_0, max_version=constants.VM_VERSION_254_0):
    """Ensures that the wrapped method's VM meets the version requirements.

    Some Hyper-V operations require a minimum VM version in order to succeed.
    For example, Production Checkpoints are supported on VM Versions 6.2 and
    newer.

    Clustering Hyper-V compute nodes may change the list of supported VM
    versions list and the default VM version on that host.

    :param min_version: string, the VM's minimum version required for the
        operation to succeed.
    :param max_version: string, the VM's maximum version required for the
        operation to succeed.
    :raises exceptions.InvalidVMVersion: if the VM's version does not meet the
        given requirements.
    """

    def wrapper(func):

        def inner(*args, **kwargs):
            all_args = inspect.getcallargs(func, *args, **kwargs)
            vmsettings = all_args['vmsettings']
            vm_version_str = getattr(vmsettings, 'Version', '4.0')
            vm_version = parse_version(vm_version_str)
            if vm_version >= parse_version(min_version) and vm_version <= parse_version(max_version):
                return func(*args, **kwargs)
            raise exceptions.InvalidVMVersion(vm_name=vmsettings.ElementName, version=vm_version_str, min_version=min_version, max_version=max_version)
        return inner
    return wrapper