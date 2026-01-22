from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def manage_proxy_server(self):
    proxy_data = self.get_existing_proxy()
    if proxy_data['enabled'] == 'no':
        if self.proxy_type == 'no_proxy':
            self.log('Proxy already disabled.')
        else:
            self.create_proxy()
            self.changed = True
    elif proxy_data['enabled'] == 'yes':
        if self.proxy_type == 'no_proxy':
            self.remove_proxy()
            self.changed = True
        else:
            modify = self.probe_proxy(proxy_data)
            if modify:
                self.update_proxy(modify)
                self.changed = True