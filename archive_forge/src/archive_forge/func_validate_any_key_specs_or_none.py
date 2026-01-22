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
def validate_any_key_specs_or_none(data, key_specs=None):
    """Validate each dict in a list matches at least 1 key_spec.

    :param data: The list of dicts to validate.
    :param key_specs: An iterable collection of key spec dicts that is used
        to check each dict in data.
    :returns: None.
    :raises InvalidInput: If any of the dicts in data do not match at least
        1 of the key_specs given.
    """
    if data is None:
        return

    def dict_validator(data_dict):
        msg = _validate_any_key_spec(data_dict, key_specs=key_specs)
        if msg:
            raise n_exc.InvalidInput(error_message=msg)
    msg = _validate_list_of_items(dict_validator, data)
    if msg:
        raise n_exc.InvalidInput(error_message=msg)