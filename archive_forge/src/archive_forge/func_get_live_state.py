from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def get_live_state(self, resource_properties):
    LOG.warning("get_live_state isn't implemented for this type of resource due to specific behaviour of cron trigger in mistral.")
    return {}