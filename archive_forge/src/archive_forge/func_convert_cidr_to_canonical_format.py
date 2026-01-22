import copy
import uuid
import netaddr
from oslo_config import cfg
from oslo_utils import strutils
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.placement import utils as pl_utils
from neutron_lib.utils import net as net_utils
def convert_cidr_to_canonical_format(value):
    """CIDR is validated and converted to canonical format.

    :param value: The CIDR which needs to be checked.
    :returns: - 'value' if 'value' is CIDR with IPv4 address,
              - CIDR with canonical IPv6 address if 'value' is IPv6 CIDR.
    :raises: InvalidInput if 'value' is None, not a valid CIDR or
        invalid IP Format.
    """
    error_message = _('%s is not in a CIDR format') % value
    try:
        cidr = netaddr.IPNetwork(value)
        return str(convert_ip_to_canonical_format(cidr.ip)) + '/' + str(cidr.prefixlen)
    except netaddr.core.AddrFormatError as e:
        raise n_exc.InvalidInput(error_message=error_message) from e