import collections
import copy
from heat.common.i18n import _
from heat.common import exception
from heat.engine import constraints
from heat.engine import parameters
from heat.engine import properties
def replace_on_change(self):
    return self._props[REPLACE_ON_CHANGE]