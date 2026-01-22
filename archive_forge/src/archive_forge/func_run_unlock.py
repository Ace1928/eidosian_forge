from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def run_unlock(self):
    """
        Checks for ADOM status, if locked, it will unlock
        """
    for adom_locked in self._locked_adoms_by_user:
        locked_user = self._locked_adoms_by_user[adom_locked]
        if locked_user == self._logged_in_user:
            self.commit_changes(adom_locked)
            self.unlock_adom(adom_locked)
            self.log('unlock adom: %s with session_id:%s' % (adom_locked, self.sid))