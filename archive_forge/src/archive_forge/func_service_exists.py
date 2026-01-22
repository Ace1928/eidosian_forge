from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import daemonize
def service_exists(self):
    rc, out, err = self.execute_command('%s list' % self.telinit_cmd)
    service_exists = False
    rex = re.compile('^\\w+\\s+%s$' % self.name)
    for line in out.splitlines():
        if rex.match(line):
            service_exists = True
            break
    if not service_exists:
        self.module.fail_json(msg='telinit could not find the requested service: %s' % self.name)