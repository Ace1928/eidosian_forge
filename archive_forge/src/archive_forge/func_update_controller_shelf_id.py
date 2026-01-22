from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def update_controller_shelf_id(self):
    """Update controller shelf tray identifier."""
    current_configuration = self.get_current_configuration()
    try:
        rc, tray = self.request('storage-systems/%s/symbol/updateTray?verboseErrorResponse=true' % self.ssid, method='POST', data={'ref': current_configuration['controller_shelf_reference'], 'trayID': self.controller_shelf_id})
    except Exception as error:
        self.module.fail_json(msg='Failed to update controller shelf identifier. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))