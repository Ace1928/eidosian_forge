import functools
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
def is_replaced(self, key):
    return self.is_active and self.cast_key_to_rule(key) in self._replaced_props