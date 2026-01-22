from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def composer_command(module, command, arguments='', options=None):
    if options is None:
        options = []
    global_command = module.params['global_command']
    if not global_command:
        options.extend(['--working-dir', "'%s'" % module.params['working_dir']])
    if module.params['executable'] is None:
        php_path = module.get_bin_path('php', True, ['/usr/local/bin'])
    else:
        php_path = module.params['executable']
    if module.params['composer_executable'] is None:
        composer_path = module.get_bin_path('composer', True, ['/usr/local/bin'])
    else:
        composer_path = module.params['composer_executable']
    cmd = '%s %s %s %s %s %s' % (php_path, composer_path, 'global' if global_command else '', command, ' '.join(options), arguments)
    return module.run_command(cmd)