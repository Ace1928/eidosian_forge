from oslo_log import log as logging
from oslo_serialization import jsonutils
import tempfile
from heat.common import auth_plugin
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import template
def needs_replace_with_prop_diff(self, changed_properties_set, after_props, before_props):
    """Needs replace based on prop_diff."""
    if self.CONTEXT in changed_properties_set and after_props.get(self.CONTEXT).get('region_name') != before_props.get(self.CONTEXT).get('region_name'):
        return True
    return False