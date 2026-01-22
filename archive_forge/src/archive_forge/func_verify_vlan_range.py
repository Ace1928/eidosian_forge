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
def verify_vlan_range(vlan_range):
    """Verify a VLAN range is valid.

    :param vlan_range: An iterable who's 0 index is the min tunnel range
        and who's 1 index is the max tunnel range.
    :returns: None if the vlan_range is valid.
    :raises: NetworkVlanRangeError if vlan_range is not valid.
    """
    for vlan_tag in vlan_range:
        if not is_valid_vlan_tag(vlan_tag):
            _raise_invalid_tag(str(vlan_tag), vlan_range)
    if vlan_range[1] < vlan_range[0]:
        raise exceptions.NetworkVlanRangeError(vlan_range=vlan_range, error=_('End of VLAN range is less than start of VLAN range'))