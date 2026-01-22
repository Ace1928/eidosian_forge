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
def create_with_template(self, child_template, user_params=None, timeout_mins=None, adopt_data=None):
    """Create the nested stack with the given template."""
    name = self.physical_resource_name()
    if timeout_mins is None:
        timeout_mins = self.stack.timeout_mins
    stack_user_project_id = self.stack.stack_user_project_id
    kwargs = self._stack_kwargs(user_params, child_template, adopt_data)
    adopt_data_str = None
    if adopt_data is not None:
        if 'environment' not in adopt_data:
            adopt_data['environment'] = kwargs['params']
        if 'template' not in adopt_data:
            if isinstance(child_template, template.Template):
                adopt_data['template'] = child_template.t
            else:
                adopt_data['template'] = child_template
        adopt_data_str = json.dumps(adopt_data)
    args = {rpc_api.PARAM_TIMEOUT: timeout_mins, rpc_api.PARAM_DISABLE_ROLLBACK: True, rpc_api.PARAM_ADOPT_STACK_DATA: adopt_data_str}
    kwargs.update({'stack_name': name, 'args': args, 'environment_files': None, 'owner_id': self.stack.id, 'user_creds_id': self.stack.user_creds_id, 'stack_user_project_id': stack_user_project_id, 'nested_depth': self._child_nested_depth(), 'parent_resource_name': self.name})
    with self.translate_remote_exceptions:
        try:
            result = self.rpc_client()._create_stack(self.context, **kwargs)
        except exception.HeatException:
            with excutils.save_and_reraise_exception():
                if adopt_data is None:
                    raw_template.RawTemplate.delete(self.context, kwargs['template_id'])
    self.resource_id_set(result['stack_id'])