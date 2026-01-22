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
def get_iface_props(iface):
    ipaddr = None
    subnet = None
    if len(iface.fixed_ips) > 0:
        ipaddr = iface.fixed_ips[0]['ip_address']
        subnet = iface.fixed_ips[0]['subnet_id']
    return {self.NETWORK_PORT: iface.port_id, self.NETWORK_ID: iface.net_id, self.NETWORK_FIXED_IP: ipaddr, self.NETWORK_SUBNET: subnet}