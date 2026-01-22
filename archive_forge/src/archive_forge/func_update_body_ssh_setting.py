from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from ansible.module_utils import six
def update_body_ssh_setting(self):
    """Configure network interface ports for remote ssh access."""
    change_required = False
    if self.interface_info['ssh'] != self.ssh:
        change_required = True
    self.body.update({'enableRemoteAccess': self.ssh})
    return change_required