from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
def diff_rules(self, existing, updated):
    for rule in updated[self.sg.SECURITY_GROUP_EGRESS]:
        rule['direction'] = 'egress'
    for rule in updated[self.sg.SECURITY_GROUP_INGRESS]:
        rule['direction'] = 'ingress'
    updated_rules = list(updated.values())
    updated_all = updated_rules[0] + updated_rules[1]
    ids_to_delete = [id for id, rule in existing.items() if rule not in updated_all]
    rules_to_create = [rule for rule in updated_all if rule not in existing.values()]
    return (ids_to_delete, rules_to_create)