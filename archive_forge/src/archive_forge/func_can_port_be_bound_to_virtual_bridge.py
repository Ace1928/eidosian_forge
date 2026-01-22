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
def can_port_be_bound_to_virtual_bridge(port):
    """Returns if port can be bound to a virtual bridge (e.g.: LB, OVS)

    :param port: (dict) A port dictionary.
    :returns: True if the port VNIC type is 'normal' or 'smart-nic'; False in
              any other case.
    """
    return port[pb.VNIC_TYPE] in [pb.VNIC_NORMAL, pb.VNIC_SMARTNIC]