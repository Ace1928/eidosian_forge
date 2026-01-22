from __future__ import absolute_import, division, print_function
import json
import multiprocessing
import threading
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import request
from ansible.module_utils._text import to_native
def check_ip_address(self, systems_found, address):
    """Determine where an E-Series storage system is available at a specific ip address."""
    for port in self.ports:
        if port == '8080':
            url = 'http://%s:%s/' % (address, port)
        else:
            url = 'https://%s:%s/' % (address, port)
        try:
            rc, about = request(url + 'devmgr/v2/storage-systems/1/about', validate_certs=False, force_basic_auth=False, ignore_errors=True)
            if about['serialNumber'] in systems_found:
                systems_found[about['serialNumber']]['api_urls'].append(url)
            else:
                systems_found.update({about['serialNumber']: {'api_urls': [url], 'label': about['name'], 'addresses': [], 'proxy_ssid': '', 'proxy_required': False}})
            break
        except Exception as error:
            try:
                rc, sa_data = request(url + 'devmgr/v2/storage-systems/1/symbol/getSAData', validate_certs=False, force_basic_auth=False, ignore_errors=True)
                if rc == 401:
                    self.module.warn('Fail over and discover any storage system without a set admin password. This will discover systems without a set password such as newly deployed storage systems. Address [%s].' % address)
                    rc, graph = request(url + 'graph', validate_certs=False, url_username='admin', url_password='', timeout=self.SEARCH_TIMEOUT)
                    sa_data = graph['sa']['saData']
                if sa_data['chassisSerialNumber'] in systems_found:
                    systems_found[sa_data['chassisSerialNumber']]['api_urls'].append(url)
                else:
                    systems_found.update({sa_data['chassisSerialNumber']: {'api_urls': [url], 'label': sa_data['storageArrayLabel'], 'addresses': [], 'proxy_ssid': '', 'proxy_required': False}})
                break
            except Exception as error:
                pass