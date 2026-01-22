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
def validate_ethertype(ethertype, valid_values=None):
    """Validates ethertype is a valid ethertype.

    :param ethertype: Ethertype to validate.
    :returns: None if data is a valid ethertype.  Otherwise a human-readable
        message indicating that the data is not a valid ethertype.
    """
    if cfg.CONF.sg_filter_ethertypes:
        os_ken_ethertypes = ether_types.__dict__
        try:
            ethertype_str = str(ethertype).upper()
            if ethertype_str == 'IPV4':
                ethertype_str = 'IP'
            ethertype_name = 'ETH_TYPE_' + ethertype_str
            ethertype = os_ken_ethertypes.get(ethertype_name, ethertype)
        except TypeError:
            pass
        try:
            if isinstance(ethertype, str):
                ethertype = int(ethertype, 16)
            if isinstance(ethertype, int) and constants.ETHERTYPE_MIN <= ethertype <= constants.ETHERTYPE_MAX:
                return None
        except ValueError:
            pass
        msg = 'Ethertype %s is not a two octet hexadecimal number or ethertype name.'
        LOG.debug(msg, ethertype)
        return _(msg) % ethertype
    else:
        if ethertype in constants.VALID_ETHERTYPES:
            return None
        valids = ','.join(map(str, constants.VALID_ETHERTYPES))
        msg_data = {'ethertype': ethertype, 'valid_ethertypes': valids}
        msg = 'Ethertype %(ethertype)s is not a valid ethertype, must be one of %(valid_ethertypes)s.'
        LOG.debug(msg, msg_data)
        return _(msg) % msg_data