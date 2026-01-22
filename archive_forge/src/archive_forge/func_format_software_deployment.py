import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_software_deployment(sd):
    if sd is None:
        return
    result = {rpc_api.SOFTWARE_DEPLOYMENT_ID: sd.id, rpc_api.SOFTWARE_DEPLOYMENT_SERVER_ID: sd.server_id, rpc_api.SOFTWARE_DEPLOYMENT_INPUT_VALUES: sd.input_values, rpc_api.SOFTWARE_DEPLOYMENT_OUTPUT_VALUES: sd.output_values, rpc_api.SOFTWARE_DEPLOYMENT_ACTION: sd.action, rpc_api.SOFTWARE_DEPLOYMENT_STATUS: sd.status, rpc_api.SOFTWARE_DEPLOYMENT_STATUS_REASON: sd.status_reason, rpc_api.SOFTWARE_DEPLOYMENT_CONFIG_ID: sd.config.id, rpc_api.SOFTWARE_DEPLOYMENT_CREATION_TIME: heat_timeutils.isotime(sd.created_at)}
    if sd.updated_at:
        result[rpc_api.SOFTWARE_DEPLOYMENT_UPDATED_TIME] = heat_timeutils.isotime(sd.updated_at)
    return result