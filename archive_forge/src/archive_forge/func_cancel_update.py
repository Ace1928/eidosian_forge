import socket
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.api.aws import exception
from heat.api.aws import utils as api_utils
from heat.common import exception as heat_exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
def cancel_update(self, req):
    action = 'CancelUpdateStack'
    self._enforce(req, action)
    con = req.context
    stack_name = req.params['StackName']
    stack_identity = self._get_identity(con, stack_name)
    try:
        self.rpc_client.stack_cancel_update(con, stack_identity=stack_identity, cancel_with_rollback=True)
    except Exception as ex:
        return exception.map_remote_error(ex)
    return api_utils.format_response(action, {})