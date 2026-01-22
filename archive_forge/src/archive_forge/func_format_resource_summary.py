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
def format_resource_summary(r):
    """Reformat engine output to AWS "StackResourceSummary" format."""
    keymap = {rpc_api.RES_UPDATED_TIME: 'LastUpdatedTimestamp', rpc_api.RES_NAME: 'LogicalResourceId', rpc_api.RES_PHYSICAL_ID: 'PhysicalResourceId', rpc_api.RES_STATUS_DATA: 'ResourceStatusReason', rpc_api.RES_TYPE: 'ResourceType'}
    result = api_utils.reformat_dict_keys(keymap, r)
    result['ResourceStatus'] = self._resource_status(r)
    return result