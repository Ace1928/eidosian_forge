import functools
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
def has_translation(self, key):
    key = self.cast_key_to_rule(key)
    return self.is_active and (key in self._rules or key in self.resolved_translations)