import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_stack_output(output_defn, resolve_value=True):
    result = {rpc_api.OUTPUT_KEY: output_defn.name, rpc_api.OUTPUT_DESCRIPTION: output_defn.description()}
    if resolve_value:
        value = None
        try:
            value = output_defn.get_value()
        except Exception as ex:
            result.update({rpc_api.OUTPUT_ERROR: str(ex)})
        finally:
            result.update({rpc_api.OUTPUT_VALUE: value})
    return result