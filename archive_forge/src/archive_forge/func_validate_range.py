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
def validate_range(data, valid_values=None):
    """Check that integer value is within a range provided.

    Test is inclusive. Allows either limit to be ignored, to allow
    checking ranges where only the lower or upper limit matter.
    It is expected that the limits provided are valid integers or
    the value None.

    :param data: The data to validate.
    :param valid_values: A list of 2 elements where element 0 is the min
        value the int data can have and element 1 is the max.
    :returns: None if the data is a valid int in the given range, otherwise
        a human readable message as to why validation failed.
    """
    min_value = valid_values[0]
    max_value = valid_values[1]
    try:
        data = int(data)
    except (ValueError, TypeError):
        msg = "'%s' is not an integer"
        LOG.debug(msg, data)
        return _(msg) % data
    if min_value is not UNLIMITED and data < min_value:
        msg_data = {'data': data, 'limit': min_value}
        msg = "'%(data)s' is too small - must be at least '%(limit)d'"
        LOG.debug(msg, msg_data)
        return _(msg) % msg_data
    if max_value is not UNLIMITED and data > max_value:
        msg_data = {'data': data, 'limit': max_value}
        msg = "'%(data)s' is too large - must be no larger than '%(limit)d'"
        LOG.debug(msg, msg_data)
        return _(msg) % msg_data