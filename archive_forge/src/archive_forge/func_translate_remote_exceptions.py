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
@excutils.exception_filter
def translate_remote_exceptions(self, ex):
    if isinstance(ex, exception.ActionInProgress) and self.stack.action == self.stack.ROLLBACK:
        return True
    class_name = reflection.get_class_name(ex, fully_qualified=False)
    if not class_name.endswith('_Remote'):
        return False
    full_message = str(ex)
    if full_message.find('\n') > -1:
        message, msg_trace = full_message.split('\n', 1)
    else:
        message = full_message
    raise exception.ResourceFailure(message, self, action=self.action)