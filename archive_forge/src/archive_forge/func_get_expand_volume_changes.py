from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_expand_volume_changes(self):
    """Expand the storage specifications for the existing thick/thin volume.

        :raise AnsibleFailJson when a thick/thin volume expansion request fails.
        :return dict: dictionary containing all the necessary values for volume expansion request
        """
    request_body = dict()
    if self.size_b < int(self.volume_detail['capacity']):
        self.module.fail_json(msg='Reducing the size of volumes is not permitted. Volume [%s]. Array [%s]' % (self.name, self.ssid))
    if self.volume_detail['thinProvisioned']:
        if self.size_b > int(self.volume_detail['capacity']):
            request_body.update(dict(sizeUnit='bytes', newVirtualSize=self.size_b))
            self.module.log('Thin volume virtual size have been expanded.')
        if self.volume_detail['expansionPolicy'] == 'automatic':
            if self.thin_volume_max_repo_size_b > int(self.volume_detail['provisionedCapacityQuota']):
                request_body.update(dict(sizeUnit='bytes', newRepositorySize=self.thin_volume_max_repo_size_b))
                self.module.log('Thin volume maximum repository size have been expanded (automatic policy).')
        elif self.volume_detail['expansionPolicy'] == 'manual':
            if self.thin_volume_repo_size_b > int(self.volume_detail['currentProvisionedCapacity']):
                change = self.thin_volume_repo_size_b - int(self.volume_detail['currentProvisionedCapacity'])
                if change < 4 * 1024 ** 3 or change > 256 * 1024 ** 3 or change % (4 * 1024 ** 3) != 0:
                    self.module.fail_json(msg='The thin volume repository increase must be between or equal to 4gb and 256gb in increments of 4gb. Attempted size [%sg].' % (self.thin_volume_repo_size_b * 1024 ** 3))
                request_body.update(dict(sizeUnit='bytes', newRepositorySize=self.thin_volume_repo_size_b))
                self.module.log('Thin volume maximum repository size have been expanded (manual policy).')
    elif self.size_b > int(self.volume_detail['capacity']):
        request_body.update(dict(sizeUnit='bytes', expansionSize=self.size_b))
        self.module.log('Volume storage capacities have been expanded.')
    return request_body