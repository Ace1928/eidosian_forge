from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
def get_reference_id(self):
    return self.resource_id