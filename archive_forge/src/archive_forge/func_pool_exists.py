from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.six import iteritems
from ansible.module_utils.basic import AnsibleModule
def pool_exists(self):
    cmd = [self.module.get_bin_path('zpool'), 'list', self.name]
    rc, dummy, dummy = self.module.run_command(cmd)
    return rc == 0