from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def get_current_properties(self):
    cmd = [self.zfs_cmd, 'get', '-H', '-p', '-o', 'property,value,source']
    if self.enhanced_sharing:
        cmd += ['-e']
    cmd += ['all', self.name]
    rc, out, err = self.module.run_command(cmd)
    properties = dict()
    for line in out.splitlines():
        prop, value, source = line.split('\t')
        if source in ('local', 'received', '-'):
            properties[prop] = value
    if self.enhanced_sharing:
        properties['sharenfs'] = properties.get('share.nfs', None)
        properties['sharesmb'] = properties.get('share.smb', None)
    return properties