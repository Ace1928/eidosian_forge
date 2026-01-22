import collections
import copy
import functools
import itertools
import math
from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import timeutils
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import support
from heat.scaling import rolling_update
from heat.scaling import template as scl_template
def get_nested_parameters_stack(self):
    """Return a nested group of size 1 for validation."""
    names = self._resource_names(1)
    child_template = self._assemble_nested(names)
    params = self.child_params()
    name = '%s-%s' % (self.stack.name, self.name)
    return self._parse_nested_stack(name, child_template, params)