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
def is_valid_geneve_vni(vni):
    """Validate a Geneve VNI

    :param vni: The VNI to validate.
    :returns: True if vni is a number that's a valid Geneve VNI.
    """
    return _is_valid_range(vni, constants.MIN_GENEVE_VNI, constants.MAX_GENEVE_VNI)