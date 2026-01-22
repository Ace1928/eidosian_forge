import collections
import functools
import inspect
import re
import netaddr
from os_ken.lib.packet import ether_types
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from webob import exc
from neutron_lib._i18n import _
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.plugins import directory
from neutron_lib.services.qos import constants as qos_consts
def validate_service_plugin_type(data, valid_values=None):
    """Validates data is a valid service plugin.

    :param data: The service plugin type to validate.
    :param valid_values: Not used.
    :returns: None if data is a valid service plugin known to the plugin
        directory.
    :raises: InvalidServiceType if data is not a service known by the
        plugin directory.
    """
    if not directory.get_plugin(data):
        raise n_exc.InvalidServiceType(service_type=data)