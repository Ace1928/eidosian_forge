from __future__ import absolute_import, division, print_function
import os
import platform
import re
import shlex
import sqlite3
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def package_latest(names, pkg_spec, module):
    if module.params['build'] is True:
        module.fail_json(msg='the combination of build=%s and state=latest is not supported' % module.params['build'])
    upgrade_cmd = 'pkg_add -um'
    if module.check_mode:
        upgrade_cmd += 'n'
    if module.params['clean']:
        upgrade_cmd += 'c'
    if module.params['quick']:
        upgrade_cmd += 'q'
    if module.params['snapshot']:
        upgrade_cmd += ' -Dsnap'
    for name in names:
        if pkg_spec[name]['installed_state'] is True:
            pkg_spec[name]['rc'], pkg_spec[name]['stdout'], pkg_spec[name]['stderr'] = execute_command('%s %s' % (upgrade_cmd, name), module)
            pkg_spec[name]['changed'] = False
            for installed_name in pkg_spec[name]['installed_names']:
                module.debug('package_latest(): checking for pre-upgrade package name: %s' % installed_name)
                match = re.search('\\W%s->.+: ok\\W' % re.escape(installed_name), pkg_spec[name]['stdout'])
                if match:
                    module.debug('package_latest(): pre-upgrade package name match: %s' % installed_name)
                    pkg_spec[name]['changed'] = True
                    break
            if pkg_spec[name]['changed'] is not True:
                if pkg_spec[name]['stderr']:
                    pkg_spec[name]['rc'] = 1
        else:
            module.debug("package_latest(): package '%s' is not installed, will be handled by package_present()" % name)
            pkg_spec['package_latest_leftovers'].append(name)
    if pkg_spec['package_latest_leftovers']:
        module.debug('package_latest(): calling package_present() to handle leftovers')
        package_present(names, pkg_spec, module)