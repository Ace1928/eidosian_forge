from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
def delete_default_egress_rules(self, sec):
    """Delete the default rules which allow all egress traffic."""
    if self.sg.properties[self.sg.SECURITY_GROUP_EGRESS]:
        for rule in sec['security_group_rules']:
            if rule['direction'] == 'egress':
                self.client.delete_security_group_rule(rule['id'])