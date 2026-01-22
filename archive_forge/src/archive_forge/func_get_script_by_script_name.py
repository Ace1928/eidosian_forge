from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_script_by_script_name(self, script_name):
    """Get script by script name

        Args:
            script_name: script name

        Returns:
            script matching script name

        """
    try:
        if script_name is None:
            return {}
        script_list = self._zapi.script.get({'output': 'extend', 'filter': {'name': [script_name]}})
        if len(script_list) < 1:
            self._module.fail_json(msg='Script not found: %s' % script_name)
        else:
            return script_list[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get script '%s': %s" % (script_name, e))