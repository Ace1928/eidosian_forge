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
def validate_ip_pools(data, valid_values=None):
    """Validate that start and end IP addresses are present.

    In addition to this the IP addresses will also be validated.

    :param data: The data to validate. Must be a list-like structure of
        IP pool dicts that each have a 'start' and 'end' key value.
    :param valid_values: Not used!
    :returns: None if data is a valid list of IP pools, otherwise a message
        indicating why the data is invalid.
    """
    if not isinstance(data, list):
        msg = "Invalid data format for IP pool: '%s'"
        LOG.debug(msg, data)
        return _(msg) % data
    expected_keys = ['start', 'end']
    for ip_pool in data:
        msg = _verify_dict_keys(expected_keys, ip_pool)
        if msg:
            return msg
        for k in expected_keys:
            msg = validate_ip_address(ip_pool[k])
            if msg:
                return msg