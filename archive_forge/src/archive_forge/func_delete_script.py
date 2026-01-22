from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def delete_script(self, script_ids):
    if self._module.check_mode:
        self._module.exit_json(changed=True)
    self._zapi.script.delete(script_ids)