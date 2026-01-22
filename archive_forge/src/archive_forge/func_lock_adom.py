from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def lock_adom(self, adom=None):
    """
        Locks an ADOM for changes
        """
    if adom:
        if adom.lower() == 'global':
            url = '/dvmdb/global/workspace/lock/'
        else:
            url = '/dvmdb/adom/{adom}/workspace/lock/'.format(adom=adom)
    else:
        url = '/dvmdb/adom/root/workspace/lock'
    code, respobj = self.send_request('exec', self._tools.format_request('exec', url))
    if code == 0 and respobj['status']['message'].lower() == 'ok':
        self.add_adom_to_lock_list(adom)
    return (code, respobj)