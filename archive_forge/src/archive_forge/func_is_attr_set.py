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
def is_attr_set(attribute):
    """Determine if an attribute value is set.

    :param attribute: The attribute value to check.
    :returns: False if the attribute value is None or ATTR_NOT_SPECIFIED,
        otherwise True.
    """
    return not (attribute is None or attribute is constants.ATTR_NOT_SPECIFIED)