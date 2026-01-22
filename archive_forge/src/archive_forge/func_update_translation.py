import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
def update_translation(self, rules, client_resolve=True, ignore_resolve_error=False):
    self.translation.set_rules(rules, client_resolve=client_resolve, ignore_resolve_error=ignore_resolve_error)