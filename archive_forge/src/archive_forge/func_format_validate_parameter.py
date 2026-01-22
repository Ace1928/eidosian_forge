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
def format_validate_parameter(key, value):
    """Reformat engine output into AWS "ValidateTemplate" format."""
    return {'ParameterKey': key, 'DefaultValue': value.get(rpc_api.PARAM_DEFAULT, ''), 'Description': value.get(rpc_api.PARAM_DESCRIPTION, ''), 'NoEcho': value.get(rpc_api.PARAM_NO_ECHO, 'false')}