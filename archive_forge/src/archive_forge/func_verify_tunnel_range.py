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
def verify_tunnel_range(tunnel_range, tunnel_type):
    """Verify a given tunnel range is valid given it's tunnel type.

    Existing validation is done for GRE, VXLAN and GENEVE types as per
    _TUNNEL_MAPPINGS.

    :param tunnel_range: An iterable who's 0 index is the min tunnel range
        and who's 1 index is the max tunnel range.
    :param tunnel_type: The tunnel type of the range.
    :returns: None if the tunnel_range is valid.
    :raises: NetworkTunnelRangeError if tunnel_range is invalid.
    """
    if tunnel_type in _TUNNEL_MAPPINGS:
        for ident in tunnel_range:
            if not _TUNNEL_MAPPINGS[tunnel_type](ident):
                raise exceptions.NetworkTunnelRangeError(tunnel_range=tunnel_range, error=_('%(id)s is not a valid %(type)s identifier') % {'id': ident, 'type': tunnel_type})
    if tunnel_range[1] < tunnel_range[0]:
        raise exceptions.NetworkTunnelRangeError(tunnel_range=tunnel_range, error=_('End of tunnel range is less than start of tunnel range'))