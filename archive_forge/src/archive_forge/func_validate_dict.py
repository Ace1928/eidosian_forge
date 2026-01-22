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
def validate_dict(data, key_specs=None):
    """Validate data is a dict optionally containing a specific set of keys.

    :param data: The data to validate.
    :param key_specs: The optional list of keys that must be contained in
        data.
    :returns: None if data is a dict and (optionally) contains only key_specs.
        Otherwise a human readable message is returned indicating why data is
        not valid.
    """
    if not isinstance(data, dict):
        msg = "'%s' is not a dictionary"
        LOG.debug(msg, data)
        return _(msg) % data
    if not key_specs:
        return
    required_keys = [key for key, spec in key_specs.items() if spec.get('required')]
    if required_keys:
        msg = _verify_dict_keys(required_keys, data, False)
        if msg:
            return msg
    unexpected_keys = [key for key in data if key not in key_specs]
    if unexpected_keys:
        msg_data = ', '.join(unexpected_keys)
        msg = 'Unexpected keys supplied: %s'
        LOG.debug(msg, msg_data)
        return _(msg) % msg_data
    for key, key_validator in [(k, v) for k, v in key_specs.items() if k in data]:
        msg = _validate_dict_item(key, key_validator, data)
        if msg:
            return msg