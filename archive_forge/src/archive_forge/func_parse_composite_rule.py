from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine import support
def parse_composite_rule(self, props):
    composite_rule = props.get(self.COMPOSITE_RULE)
    operator = composite_rule[self.COMPOSITE_OPERATOR]
    rules = composite_rule[self.RULES]
    props[self.COMPOSITE_RULE] = {operator: rules}