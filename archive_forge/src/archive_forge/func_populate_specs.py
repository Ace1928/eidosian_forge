from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def populate_specs(self):
    self.service_locator.instanceUuid = self.destination_content.about.instanceUuid
    self.service_locator.url = 'https://' + self.destination_vcenter + ':' + str(self.params['port']) + '/sdk'
    if not self.destination_vcenter_validate_certs:
        self.service_locator.sslThumbprint = self.get_cert_fingerprint(self.destination_vcenter, self.destination_vcenter_port, self.module.params['proxy_host'], self.module.params['proxy_port'])
    creds = vim.ServiceLocatorNamePassword()
    creds.username = self.destination_vcenter_username
    creds.password = self.destination_vcenter_password
    self.service_locator.credential = creds
    self.relocate_spec.datastore = self.destination_datastore
    self.relocate_spec.pool = self.destination_resource_pool
    self.relocate_spec.service = self.service_locator
    self.relocate_spec.host = self.destination_host
    self.clone_spec.config = self.config_spec
    self.clone_spec.powerOn = True if self.params['state'].lower() == 'poweredon' else False
    self.clone_spec.location = self.relocate_spec
    self.clone_spec.template = self.params['is_template']