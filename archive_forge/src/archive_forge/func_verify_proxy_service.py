from __future__ import absolute_import, division, print_function
import json
import multiprocessing
import threading
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import request
from ansible.module_utils._text import to_native
def verify_proxy_service(self):
    """Verify proxy url points to a web services proxy."""
    try:
        rc, about = request(self.proxy_about_url, validate_certs=self.proxy_validate_certs)
        if not about['runningAsProxy']:
            self.module.fail_json(msg='Web Services is not running as a proxy!')
    except Exception as error:
        self.module.fail_json(msg='Proxy is not available! Check proxy_url. Error [%s].' % to_native(error))