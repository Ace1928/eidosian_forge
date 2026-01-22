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
def convert_uppercase_ip(data):
    """Uppercase "ip" if present at start of data case-insensitive

    Can be used for instance to accept both "ipv4" and "IPv4".

    :param data: The data to convert
    :returns: if data is a string starting with "ip" case insensitive, then
              the return value is data with the first two letter uppercased
    """
    return convert_prefix_forced_case(data, 'IP')