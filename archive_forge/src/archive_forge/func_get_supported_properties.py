from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def get_supported_properties(self, setting):
    properties = []
    if setting == '802-11-wireless-security':
        set_property = 'psk'
        set_value = 'FAKEVALUE'
        commands = ['set %s.%s %s' % (setting, set_property, set_value)]
    else:
        commands = []
    commands += ['print %s' % setting, 'quit', 'yes']
    rc, out, err = self.execute_edit_commands(commands, arguments=['type', self.type])
    if rc != 0:
        raise NmcliModuleError(err)
    for line in out.splitlines():
        prefix = '%s.' % setting
        if line.startswith(prefix):
            pair = line.split(':', 1)
            property = pair[0].strip().replace(prefix, '')
            properties.append(property)
    return properties