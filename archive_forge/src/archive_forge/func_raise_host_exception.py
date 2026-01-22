from __future__ import (absolute_import, division, print_function)
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def raise_host_exception(self):
    events = self._connection.system_service().events_service().list(from_=int(self.start_event.index))
    error_events = [event.description for event in events if event.host is not None and (event.host.id == self.param('id') or event.host.name == self.param('name')) and (event.severity in [otypes.LogSeverity.WARNING, otypes.LogSeverity.ERROR])]
    if error_events:
        raise Exception('Error message: %s' % error_events)
    return True