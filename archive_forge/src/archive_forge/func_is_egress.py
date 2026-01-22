from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
def is_egress(rule):
    return rule[self.RULE_DIRECTION] == 'egress'