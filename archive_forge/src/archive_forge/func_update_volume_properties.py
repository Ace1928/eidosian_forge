from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def update_volume_properties(self):
    """Update existing thin-volume or volume properties.

        :raise AnsibleFailJson when either thick/thin volume update request fails.
        :return bool: whether update was applied
        """
    self.wait_for_volume_availability()
    self.volume_detail = self.get_volume()
    request_body = self.get_volume_property_changes()
    if request_body:
        if self.thin_provision:
            try:
                rc, resp = self.request('storage-systems/%s/thin-volumes/%s' % (self.ssid, self.volume_detail['id']), data=request_body, method='POST')
            except Exception as error:
                self.module.fail_json(msg='Failed to update thin volume properties. Volume [%s]. Array Id [%s]. Error[%s].' % (self.name, self.ssid, to_native(error)))
        else:
            try:
                rc, resp = self.request('storage-systems/%s/volumes/%s' % (self.ssid, self.volume_detail['id']), data=request_body, method='POST')
            except Exception as error:
                self.module.fail_json(msg='Failed to update volume properties. Volume [%s]. Array Id [%s]. Error[%s].' % (self.name, self.ssid, to_native(error)))
        return True
    return False