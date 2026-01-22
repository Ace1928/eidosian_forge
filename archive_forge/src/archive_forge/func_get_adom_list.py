from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def get_adom_list(self):
    """
        Gets the list of ADOMs for the FortiManager
        """
    if self._uses_adoms:
        url = '/dvmdb/adom'
        rc, resp_obj = self.send_request('get', self._tools.format_request('get', url))
        if rc != 0:
            err_msg = 'An error occurred trying to get the ADOM Info. Error %s: %s' % (rc, to_text(resp_obj))
            raise FMGBaseException(msg=err_msg)
        else:
            append_list = ['root', 'global']
            for adom in resp_obj['data']:
                if adom['tab_status'] != '':
                    append_list.append(to_text(adom['name']))
            self._adom_list = append_list
        self.log('adom list: %s' % str(self._adom_list))
        return (rc, resp_obj)