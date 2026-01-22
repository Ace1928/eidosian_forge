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
def convert_to_mac_if_none(data):
    """Convert to a random mac address if data is None

    :param data: The data value
    :return: Random mac address if data is None, else return data.
    """
    if data is None:
        return net_utils.get_random_mac(cfg.CONF.base_mac.split(':'))
    return data