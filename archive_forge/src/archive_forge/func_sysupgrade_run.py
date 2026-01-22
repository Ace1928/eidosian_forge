from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
def sysupgrade_run(module):
    sysupgrade_bin = module.get_bin_path('/usr/sbin/sysupgrade', required=True)
    cmd = [sysupgrade_bin]
    changed = False
    warnings = []
    if module.params['snapshot']:
        run_flag = ['-s']
        if module.params['force']:
            run_flag.append('-f')
    else:
        run_flag = ['-r']
    if module.params['keep_files']:
        run_flag.append('-k')
    if module.params['fetch_only']:
        run_flag.append('-n')
    if module.params['installurl']:
        run_flag.append(module.params['installurl'])
    rc, out, err = module.run_command(cmd + run_flag)
    if rc != 0:
        module.fail_json(msg='Command %s failed rc=%d, out=%s, err=%s' % (cmd, rc, out, err))
    elif out.lower().find('already on latest snapshot') >= 0:
        changed = False
    elif out.lower().find('upgrade on next reboot') >= 0:
        changed = True
    return dict(changed=changed, rc=rc, stderr=err, stdout=out, warnings=warnings)