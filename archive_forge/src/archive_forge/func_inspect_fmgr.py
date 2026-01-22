from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def inspect_fmgr(self):
    rc, status = self.get_system_status()
    if rc == -11:
        self.logout()
        raise FMGBaseException(msg='Error -11 -- the Session ID was likely malformed somehow. Exiting')
    elif rc == 0:
        try:
            self.check_mode()
            if self._uses_adoms:
                self.get_adom_list()
            if self._uses_workspace:
                self.get_locked_adom_list()
            self._connected_fmgr = status
        except Exception as e:
            self.log('inspect_fmgr exception: %s' % e)