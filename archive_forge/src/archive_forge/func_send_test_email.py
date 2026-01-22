from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def send_test_email(self):
    """Send a test email to verify that the provided configuration is valid and functional."""
    if not self.check_mode:
        try:
            rc, result = request(self.url + 'storage-systems/%s/device-alerts/alert-email-test' % self.ssid, timeout=300, method='POST', headers=HEADERS, **self.creds)
            if result['response'] != 'emailSentOK':
                self.module.fail_json(msg='The test email failed with status=[%s]! Array Id [%s].' % (result['response'], self.ssid))
        except Exception as err:
            self.module.fail_json(msg='We failed to send the test email! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))