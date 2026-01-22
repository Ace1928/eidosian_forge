import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
@cached_attrs.setter
def cached_attrs(self, c_attrs):
    if c_attrs is None:
        self._resolved_values = {}
    else:
        self._resolved_values = c_attrs
    self._has_new_resolved = False