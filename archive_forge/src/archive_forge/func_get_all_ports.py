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
@staticmethod
def get_all_ports(server):
    return itertools.chain(server._data_get_ports(), server._data_get_ports('external_ports'))