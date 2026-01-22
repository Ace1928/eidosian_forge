from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback
def ensure_powered_on(self, wait=True, wait_timeout=300):
    if self.is_powered_on():
        return
    if self.status == 'off':
        self.power_on()
    if wait:
        end_time = time.monotonic() + wait_timeout
        while time.monotonic() < end_time:
            time.sleep(10)
            self.update_attr()
            if self.is_powered_on():
                if not self.ip_address:
                    raise TimeoutError('No ip is found.', self.id)
                return
        raise TimeoutError('Wait for droplet running timeout', self.id)