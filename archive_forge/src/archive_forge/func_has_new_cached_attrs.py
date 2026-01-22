import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
def has_new_cached_attrs(self):
    """Returns True if cached_attrs have changed

        Allows the caller to determine if this instance's cached_attrs
        have been updated since they were initially set (if at all).
        """
    return self._has_new_resolved