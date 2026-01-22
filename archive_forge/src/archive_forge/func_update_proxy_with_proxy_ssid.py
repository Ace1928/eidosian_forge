from __future__ import absolute_import, division, print_function
import json
import multiprocessing
import threading
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import request
from ansible.module_utils._text import to_native
def update_proxy_with_proxy_ssid(self):
    """Determine the current proxy ssid for all discovered-proxy_required storage systems."""
    systems = []
    try:
        rc, systems = request(self.proxy_url + 'storage-systems', validate_certs=self.proxy_validate_certs, force_basic_auth=True, url_username=self.proxy_username, url_password=self.proxy_password)
    except Exception as error:
        self.module.fail_json(msg='Failed to ascertain storage systems added to Web Services Proxy.')
    for system_key, system_info in self.systems_found.items():
        if self.systems_found[system_key]['proxy_required']:
            for system in systems:
                if system_key == system['chassisSerialNumber']:
                    self.systems_found[system_key]['proxy_ssid'] = system['id']