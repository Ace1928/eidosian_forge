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
def validate_nested_stack(self):
    if not self.get_size():
        return
    first_name = next(self._resource_names())
    test_tmpl = self._assemble_nested([first_name], include_all=True)
    res_def = next(iter(test_tmpl.resource_definitions(None).values()))
    self.stack.env.get_class_to_instantiate(res_def.resource_type)
    try:
        name = '%s-%s' % (self.stack.name, self.name)
        nested_stack = self._parse_nested_stack(name, test_tmpl, self.child_params())
        nested_stack.strict_validate = False
        nested_stack.validate()
    except Exception as ex:
        path = '%s<%s>' % (self.name, self.template_url)
        raise exception.StackValidationFailed(ex, path=[self.stack.t.RESOURCES, path])