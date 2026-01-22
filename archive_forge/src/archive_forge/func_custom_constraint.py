import collections
import json
import numbers
import re
from oslo_cache import core
from oslo_config import cfg
from oslo_log import log
from oslo_utils import reflection
from oslo_utils import strutils
from heat.common import cache
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resources
@property
def custom_constraint(self):
    if self._custom_constraint is None:
        if self._environment is None:
            self._environment = resources.global_env()
        constraint_class = self._environment.get_constraint(self.name)
        if constraint_class:
            self._custom_constraint = constraint_class()
    return self._custom_constraint