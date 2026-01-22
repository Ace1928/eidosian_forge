from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_action_by_name(self, name):
    """Get action by name

        Args:
            name: Name of the action.

        Returns:
            dict: Zabbix action

        """
    try:
        action_list = self._zapi.action.get({'output': 'extend', 'filter': {'name': [name]}})
        if len(action_list) < 1:
            self._module.fail_json(msg='Action not found: %s' % name)
        else:
            return action_list[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get ID of '%s': %s" % (name, e))