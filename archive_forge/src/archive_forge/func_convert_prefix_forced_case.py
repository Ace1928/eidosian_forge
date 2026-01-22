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
def convert_prefix_forced_case(data, prefix):
    """If <prefix> is a prefix of data, case insensitive, then force its case

    This converter forces the case of a given prefix of a string.

    Example, with prefix="Foo":
    * 'foobar' converted into 'Foobar'
    * 'fOozar' converted into 'Foozar'
    * 'FOObaz' converted into 'Foobaz'

    :param data: The data to convert
    :returns: if data is a string starting with <prefix> in a case insensitive
              comparison, then the return value is data with this prefix
              replaced by <prefix>
    """
    plen = len(prefix)
    if isinstance(data, str) and len(data) >= plen and (data[0:plen].lower() == prefix.lower()):
        return prefix + data[plen:]
    return data