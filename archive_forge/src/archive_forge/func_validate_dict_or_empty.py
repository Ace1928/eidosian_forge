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
def validate_dict_or_empty(data, key_specs=None):
    """Validate data is {} or a dict containing a specific set of keys.

    :param data: The data to validate.
    :param key_specs: The optional list of keys that must be contained in
        data.
    :returns: None if data is {} or a dict (optionally) containing
        only key_specs. Otherwise a human readable message is returned
        indicating why data is not valid.
    """
    if data != {}:
        return validate_dict(data, key_specs)