from heat.common.i18n import _
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine import support
def get_alarm_props(self, props):
    kwargs = self.actions_to_urls(props)
    kwargs = self._reformat_properties(kwargs)
    return kwargs