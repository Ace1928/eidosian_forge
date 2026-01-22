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
def validate_not_empty_string(data, max_len=None):
    """Validate data is a non-empty/non-blank string.

    :param data: The data to validate.
    :param max_len: An optional cap on the length of the string data.
    :returns: None if the data is non-empty/non-blank, otherwise a human
        readable string message indicating why validation failed.
    """
    msg = validate_string(data, max_len=max_len)
    if msg:
        return msg
    if not data.strip():
        msg = "'%s' Blank strings are not permitted"
        LOG.debug(msg, data)
        return _(msg) % data