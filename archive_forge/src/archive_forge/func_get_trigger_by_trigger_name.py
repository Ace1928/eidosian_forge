from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_trigger_by_trigger_name(self, trigger_name):
    """Get trigger by trigger name

        Args:
            trigger_name: trigger name.

        Returns:
            trigger matching trigger name

        """
    try:
        trigger_list = self._zapi.trigger.get({'output': 'extend', 'filter': {'description': [trigger_name]}})
        if len(trigger_list) < 1:
            self._module.fail_json(msg='Trigger not found: %s' % trigger_name)
        else:
            return trigger_list[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get trigger '%s': %s" % (trigger_name, e))