from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def process_workspace_locking_internal(self, param):
    if not self._uses_workspace or not self._logged:
        return
    if 'workspace_locking_adom' not in param or not param['workspace_locking_adom']:
        return
    adom_to_lock = param['workspace_locking_adom']
    adom_to_lock_timeout = param['workspace_locking_timeout']
    self.log('trying to acquire lock for adom: %s within %s seconds by user: %s' % (adom_to_lock, adom_to_lock_timeout, self._logged_in_user))
    if adom_to_lock in self._locked_adoms_by_user:
        if self._locked_adoms_by_user[adom_to_lock] == self._logged_in_user:
            self.log('adom: %s has already been acquired by user: %s' % (adom_to_lock, self._logged_in_user))
        else:
            total_wait_time = 0
            while total_wait_time < adom_to_lock_timeout:
                code, resp_obj = self.lock_adom(adom_to_lock)
                self.log('waiting adom:%s lock to be released by %s, total time spent:%s seconds status:%s' % (adom_to_lock, self._locked_adoms_by_user[adom_to_lock], total_wait_time, 'success' if code == 0 else 'failure'))
                if code == 0:
                    self._locked_adoms_by_user[adom_to_lock] = self._logged_in_user
                    break
                time.sleep(5)
                total_wait_time += 5
    else:
        code, resp_obj = self.lock_adom(adom_to_lock)
        self.log('adom:%s locked by user: %s status:%s' % (adom_to_lock, self._logged_in_user, 'success' if code == 0 else 'failure'))
        if code == 0:
            self._locked_adoms_by_user[adom_to_lock] = self._logged_in_user