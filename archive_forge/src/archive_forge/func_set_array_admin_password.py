from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def set_array_admin_password(self):
    """Set the array's admin password."""
    if self.is_proxy():
        if self.ssid == '0' or self.ssid.lower() == 'proxy':
            self.creds['url_password'] = 'admin'
            try:
                body = {'currentAdminPassword': '', 'updates': {'userName': 'admin', 'newPassword': self.password}}
                rc, proxy = self.request('local-users', method='POST', data=body)
            except Exception as error:
                self.creds['url_password'] = ''
                try:
                    body = {'currentAdminPassword': '', 'updates': {'userName': 'admin', 'newPassword': self.password}}
                    rc, proxy = self.request('local-users', method='POST', data=body)
                except Exception as error:
                    self.module.fail_json(msg="Failed to set proxy's admin password. Error [%s]." % to_native(error))
            self.creds['url_password'] = self.password
        else:
            try:
                body = {'currentAdminPassword': '', 'newPassword': self.password, 'adminPassword': True}
                rc, storage_system = self.request('storage-systems/%s/passwords' % self.ssid, method='POST', data=body)
            except Exception as error:
                self.module.fail_json(msg="Failed to set storage system's admin password. Array [%s]. Error [%s]." % (self.ssid, to_native(error)))
    else:
        self.creds['url_password'] = ''
        try:
            body = {'currentAdminPassword': '', 'updates': {'userName': 'admin', 'newPassword': self.password}}
            rc, proxy = self.request('storage-systems/%s/local-users' % self.ssid, method='POST', data=body)
        except Exception as error:
            self.module.fail_json(msg="Failed to set embedded storage system's admin password. Array [%s]. Error [%s]." % (self.ssid, to_native(error)))
        self.creds['url_password'] = self.password