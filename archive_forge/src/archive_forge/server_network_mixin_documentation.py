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
Store in resource's data IDs of ports created by nova for server.

        If no port property is specified and no internal port has been created,
        nova client takes no port-id and calls port creating into server
        creating. We need to store information about that ports, so store
        their IDs to data with key `external_ports`.
        