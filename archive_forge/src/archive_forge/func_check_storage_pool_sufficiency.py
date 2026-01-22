from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def check_storage_pool_sufficiency(self):
    """Perform a series of checks as to the sufficiency of the storage pool for the volume."""
    if not self.pool_detail:
        self.module.fail_json(msg='Requested storage pool (%s) not found' % self.storage_pool_name)
    if not self.volume_detail:
        if self.thin_provision and (not self.pool_detail['diskPool']):
            self.module.fail_json(msg='Thin provisioned volumes can only be created on raid disk pools.')
        if self.data_assurance_enabled and (not (self.pool_detail['protectionInformationCapabilities']['protectionInformationCapable'] and self.pool_detail['protectionInformationCapabilities']['protectionType'] == 'type2Protection')):
            self.module.fail_json(msg='Data Assurance (DA) requires the storage pool to be DA-compatible. Array [%s].' % self.ssid)
        if int(self.pool_detail['freeSpace']) < self.size_b and (not self.thin_provision):
            self.module.fail_json(msg="Not enough storage pool free space available for the volume's needs. Array [%s]." % self.ssid)
    elif int(self.pool_detail['freeSpace']) < int(self.volume_detail['totalSizeInBytes']) - self.size_b and (not self.thin_provision):
        self.module.fail_json(msg="Not enough storage pool free space available for the volume's needs. Array [%s]." % self.ssid)