from oslo_log import log as logging
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
@staticmethod
def prepare_trunk_remove_subports_body(subports):
    """Prepares body for PUT /v2.0/trunks/TRUNK_ID/remove_subports."""
    return {'sub_ports': [{'port_id': sp['port']} for sp in subports]}