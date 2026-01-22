from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def delete_template_group(self, group_ids):
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        self._zapi.templategroup.delete(group_ids)
    except Exception as e:
        self._module.fail_json(msg='Failed to delete template group(s), Exception: %s' % e)