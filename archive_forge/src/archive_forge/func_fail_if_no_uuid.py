from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def fail_if_no_uuid(self):
    """Prevent a logic error."""
    if self.app_uuid is None:
        msg = 'function should not be called before application uuid is set.'
        return (None, msg)
    return (None, None)