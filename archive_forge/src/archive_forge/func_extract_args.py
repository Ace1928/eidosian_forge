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
def extract_args(params):
    """Extract request params and reformat them to match engine API.

            FIXME: we currently only support a subset of
            the AWS defined parameters (both here and in the engine)
            """
    keymap = {'TimeoutInMinutes': rpc_api.PARAM_TIMEOUT, 'DisableRollback': rpc_api.PARAM_DISABLE_ROLLBACK}
    if 'DisableRollback' in params and 'OnFailure' in params:
        msg = _('DisableRollback and OnFailure may not be used together')
        raise exception.HeatInvalidParameterCombinationError(detail=msg)
    result = {}
    for k in keymap:
        if k in params:
            result[keymap[k]] = params[k]
    if 'OnFailure' in params:
        value = params['OnFailure']
        if value == 'DO_NOTHING':
            result[rpc_api.PARAM_DISABLE_ROLLBACK] = 'true'
        elif value in ('ROLLBACK', 'DELETE'):
            result[rpc_api.PARAM_DISABLE_ROLLBACK] = 'false'
    return result