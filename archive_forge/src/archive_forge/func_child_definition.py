import json
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
def child_definition(self, child_template=None, user_params=None, nested_identifier=None):
    if user_params is None:
        user_params = self.child_params()
    if child_template is None:
        child_template = self.child_template()
    if nested_identifier is None:
        nested_identifier = self.nested_identifier()
    child_env = environment.get_child_environment(self.stack.env, user_params, child_resource_name=self.name, item_to_remove=self.resource_info)
    parsed_template = self._child_parsed_template(child_template, child_env)
    return stk_defn.StackDefinition(self.context, parsed_template, nested_identifier, None)