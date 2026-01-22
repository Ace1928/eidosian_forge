from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def fail_if_uuid(self, fname):
    """Prevent a logic error."""
    if self.app_uuid is not None:
        msg = 'function %s should not be called when application uuid is set: %s.' % (fname, self.app_uuid)
        return (None, msg)
    return (None, None)