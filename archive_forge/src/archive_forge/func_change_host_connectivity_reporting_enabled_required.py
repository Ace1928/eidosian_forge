from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def change_host_connectivity_reporting_enabled_required(self):
    """Determine whether host connectivity reporting state change is required."""
    if self.host_connectivity_reporting_enabled is None:
        return False
    current_configuration = self.get_current_configuration()
    return self.host_connectivity_reporting_enabled != current_configuration['host_connectivity_reporting_enabled']