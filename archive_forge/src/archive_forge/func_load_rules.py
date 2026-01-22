from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import opts
from oslo_policy import policy
from oslo_utils import excutils
from heat.common import exception
from heat.common.i18n import _
from heat import policies
def load_rules(self, force_reload=False):
    """Set the rules found in the json file on disk."""
    self.enforcer.load_rules(force_reload)