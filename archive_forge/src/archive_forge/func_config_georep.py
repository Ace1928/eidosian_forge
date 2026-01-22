from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def config_georep(self):
    if self.action != 'config':
        return ''
    options = ['gluster_log_file', 'gluster_log_level', 'log_file', 'log_level', 'changelog_log_level', 'ssh_command', 'rsync_command', 'use_tarssh', 'volume_id', 'timeout', 'sync_jobs', 'ignore_deletes', 'checkpoint', 'sync_acls', 'sync_xattrs', 'log_rsync_performance', 'rsync_options', 'use_meta_volume', 'meta_volume_mnt']
    configs = []
    for opt in options:
        value = self._validated_params(opt)
        if value:
            if value == 'reset':
                configs.append("'!" + opt.replace('_', '-') + "'")
            configs.append(opt.replace('_', '-') + ' ' + value)
    if configs:
        return configs
    value = self._validated_params('config')
    op = self._validated_params('op')
    return value + ' ' + op