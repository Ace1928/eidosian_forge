import collections
import contextlib
import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from webob import exc as web_exc
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib.api.definitions import network as net_apidef
from neutron_lib.api.definitions import port as port_apidef
from neutron_lib.api.definitions import portbindings as pb
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib.api.definitions import subnet as subnet_apidef
from neutron_lib import constants
from neutron_lib import exceptions
def parse_network_vlan_ranges(network_vlan_ranges_cfg_entries):
    """Parse a list of well formed network VLAN range string.

    Behaves like parse_network_vlan_range, but parses a list of
    network VLAN strings into an ordered dict.

    :param network_vlan_ranges_cfg_entries: The list of network VLAN
        strings to parse.
    :returns: An OrderedDict who's keys are network names and values are
        the list of VLAN ranges parsed.
    :raises: See parse_network_vlan_range.
    """
    networks = collections.OrderedDict()
    for entry in network_vlan_ranges_cfg_entries:
        network, vlan_range = parse_network_vlan_range(entry)
        if vlan_range:
            if networks.get(network) == [constants.VLAN_VALID_RANGE]:
                continue
            networks.setdefault(network, []).append(vlan_range)
        else:
            networks[network] = [constants.VLAN_VALID_RANGE]
    return networks