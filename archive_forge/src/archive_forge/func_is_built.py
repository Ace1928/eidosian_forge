from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
@staticmethod
def is_built(attributes):
    status = attributes['status']
    if status == 'BUILD':
        return False
    if status in ('ACTIVE', 'DOWN'):
        return True
    elif status in ('ERROR', 'DEGRADED'):
        raise exception.ResourceInError(resource_status=status)
    else:
        raise exception.ResourceUnknownStatus(resource_status=status, result=_('Resource is not built'))