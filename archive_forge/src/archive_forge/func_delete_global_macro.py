from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def delete_global_macro(self, global_macro_obj, macro_name):
    global_macro_id = global_macro_obj['globalmacroid']
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        self._zapi.usermacro.deleteglobal([global_macro_id])
        self._module.exit_json(changed=True, result='Successfully deleted global macro %s' % macro_name)
    except Exception as e:
        self._module.fail_json(msg='Failed to delete global macro %s: %s' % (macro_name, e))