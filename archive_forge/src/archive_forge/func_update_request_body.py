from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from ansible.module_utils import six
def update_request_body(self):
    """Verify all required changes have been made."""
    self.update_target_interface_info()
    self.body = {'controllerRef': self.get_controllers()[self.controller]['controllerRef'], 'interfaceRef': self.interface_info['id']}
    change_required = False
    if self.enable_interface is not None:
        change_required = self.update_body_enable_interface_setting()
    if self.config_method is not None:
        change_required = self.update_body_interface_settings() or change_required
    if self.dns_config_method is not None:
        change_required = self.update_body_dns_server_settings() or change_required
    if self.ntp_config_method is not None:
        change_required = self.update_body_ntp_server_settings() or change_required
    if self.ssh is not None:
        change_required = self.update_body_ssh_setting() or change_required
    self.module.log('update_request_body change_required: %s' % change_required)
    return change_required