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
def validate_subnet_service_types(service_types, valid_values=None):
    if service_types:
        if not isinstance(service_types, list):
            raise exc.HTTPBadRequest(_('Subnet service types must be a list.'))
        prefixes = list(constants.DEVICE_OWNER_PREFIXES)
        prefixes += constants.DEVICE_OWNER_COMPUTE_PREFIX
        for service_type in service_types:
            if not isinstance(service_type, str):
                raise n_exc.InvalidInputSubnetServiceType(service_type=service_type)
            if not service_type.startswith(tuple(prefixes)):
                raise n_exc.InvalidSubnetServiceType(service_type=service_type)