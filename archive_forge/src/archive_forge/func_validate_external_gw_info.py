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
def validate_external_gw_info(data, valid_values=None):
    """Validate data is an external_gateway_info.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if valid, error string otherwise.
    """
    from neutron_lib.api import converters
    return validate_dict_or_nodata(data, key_specs={'network_id': {'type:uuid': None, 'required': True}, 'external_fixed_ips': {'type:fixed_ips': None, 'required': False}, 'enable_snat': {'type:boolean': None, 'required': False, 'convert_to': converters.convert_to_boolean}, qos_consts.QOS_POLICY_ID: {'type:uuid_or_none': None, 'required': False}})