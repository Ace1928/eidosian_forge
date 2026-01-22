import itertools
import eventlet
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from heat.engine.resources.openstack.neutron import port as neutron_port
def update_floating_ip_association(self, floating_ip, flip_associate):
    if flip_associate.get('port_id'):
        self._floating_ip_neutron_associate(floating_ip, flip_associate)