from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import os
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import platform
def get_current_state(self, command):
    """ Returns the list of all app IDs; command can either be 'list' or 'outdated' """
    rc, raw_apps, err = self.run([command])
    rows = raw_apps.split('\n')
    if rows[0] == 'No installed apps found':
        rows = []
    apps = []
    for r in rows:
        r = r.split(' ', 1)
        if len(r) == 2:
            apps.append(int(r[0]))
    return apps