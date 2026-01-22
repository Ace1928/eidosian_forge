import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def format_hash(data):
    if data:
        return '\n'.join(('{}={}'.format(k, v) for k, v in data.items()))
    return None