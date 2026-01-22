from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def get_installed_versions(module, remote=False):
    cmd = get_rubygems_path(module)
    cmd.append('query')
    cmd.extend(common_opts(module))
    if remote:
        cmd.append('--remote')
        if module.params['repository']:
            cmd.extend(['--source', module.params['repository']])
    cmd.append('-n')
    cmd.append('^%s$' % module.params['name'])
    environ = get_rubygems_environ(module)
    rc, out, err = module.run_command(cmd, environ_update=environ, check_rc=True)
    installed_versions = []
    for line in out.splitlines():
        match = re.match('\\S+\\s+\\((?:default: )?(.+)\\)', line)
        if match:
            versions = match.group(1)
            for version in versions.split(', '):
                installed_versions.append(version.split()[0])
    return installed_versions