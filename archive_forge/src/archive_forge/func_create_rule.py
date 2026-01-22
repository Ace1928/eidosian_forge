from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
def create_rule(self, rule):
    try:
        self.client.create_security_group_rule({'security_group_rule': self._convert_to_neutron_rule(rule)})
    except Exception as ex:
        if not self.plugin.is_conflict(ex):
            raise