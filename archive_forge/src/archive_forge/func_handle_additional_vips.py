import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def handle_additional_vips(vips, client_manager):
    additional_vips = []
    for vip in vips:
        vip_dict = {}
        parts = vip.split(',')
        for part in parts:
            k, v = part.split('=')
            vip_dict[k.replace('-', '_')] = v
        validate_vip_dict(vip_dict, client_manager)
        additional_vips.append(vip_dict)
    return additional_vips