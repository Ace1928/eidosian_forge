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
def format_stack_summary(s):
    """Reformat engine output into the AWS "StackSummary" format."""
    keymap = {rpc_api.STACK_CREATION_TIME: 'CreationTime', rpc_api.STACK_UPDATED_TIME: 'LastUpdatedTime', rpc_api.STACK_ID: 'StackId', rpc_api.STACK_NAME: 'StackName', rpc_api.STACK_STATUS_DATA: 'StackStatusReason', rpc_api.STACK_TMPL_DESCRIPTION: 'TemplateDescription'}
    result = api_utils.reformat_dict_keys(keymap, s)
    action = s[rpc_api.STACK_ACTION]
    status = s[rpc_api.STACK_STATUS]
    result['StackStatus'] = '_'.join((action, status))
    if rpc_api.STACK_DELETION_TIME in s:
        result['DeletionTime'] = s[rpc_api.STACK_DELETION_TIME]
    return self._id_format(result)