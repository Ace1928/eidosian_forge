from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def upgrade_packages(module, slackpkg_path, packages):
    install_c = 0
    for package in packages:
        if not module.check_mode:
            rc, out, err = module.run_command('%s -default_answer=y -batch=on                                               upgrade %s' % (slackpkg_path, package))
        if not module.check_mode and (not query_package(module, slackpkg_path, package)):
            module.fail_json(msg='failed to install %s: %s' % (package, out), stderr=err)
        install_c += 1
    if install_c > 0:
        module.exit_json(changed=True, msg='present %s package(s)' % install_c)
    module.exit_json(changed=False, msg='package(s) already present')