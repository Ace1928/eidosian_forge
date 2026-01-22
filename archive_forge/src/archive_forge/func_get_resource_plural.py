from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
def get_resource_plural(self):
    """Return the plural of resource type.

        The default implementation is to return self.entity + 's',
        the rule is not appropriate for some special resources,
        e.g. qos_policy, this method should be overridden by the
        special resources if needed.
        """
    if not self.entity:
        return
    return self.entity + 's'