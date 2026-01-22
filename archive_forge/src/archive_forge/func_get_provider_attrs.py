import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_provider_attrs(parsed_args):
    attr_map = {'provider': ('provider_name', str), 'description': ('description', str), 'flavor': ('flavor', bool), 'availability_zone': ('availability_zone', bool)}
    return _map_attrs(vars(parsed_args), attr_map)