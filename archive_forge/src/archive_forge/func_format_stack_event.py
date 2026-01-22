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
def format_stack_event(e):
    """Reformat engine output into AWS "StackEvent" format."""
    keymap = {rpc_api.EVENT_ID: 'EventId', rpc_api.EVENT_RES_NAME: 'LogicalResourceId', rpc_api.EVENT_RES_PHYSICAL_ID: 'PhysicalResourceId', rpc_api.EVENT_RES_PROPERTIES: 'ResourceProperties', rpc_api.EVENT_RES_STATUS_DATA: 'ResourceStatusReason', rpc_api.EVENT_RES_TYPE: 'ResourceType', rpc_api.EVENT_STACK_ID: 'StackId', rpc_api.EVENT_STACK_NAME: 'StackName', rpc_api.EVENT_TIMESTAMP: 'Timestamp'}
    result = api_utils.reformat_dict_keys(keymap, e)
    action = e[rpc_api.EVENT_RES_ACTION]
    status = e[rpc_api.EVENT_RES_STATUS]
    result['ResourceStatus'] = '_'.join((action, status))
    result['ResourceProperties'] = jsonutils.dumps(result['ResourceProperties'])
    return self._id_format(result)