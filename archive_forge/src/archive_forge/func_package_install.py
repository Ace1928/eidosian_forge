from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def package_install(module, state, pkgs, site, update_catalog, force):
    cmd = ['pkgutil']
    if module.check_mode:
        cmd.append('-n')
    cmd.append('-iy')
    if update_catalog:
        cmd.append('-U')
    if site is not None:
        cmd.extend(['-t', site])
    if force:
        cmd.append('-f')
    cmd.extend(pkgs)
    return run_command(module, cmd)