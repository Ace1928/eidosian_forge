from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_if_usergroup_exists(self, name):
    """Check if user group exists.

        Parameters:
            name: Name of the user group.

        Returns:
            The return value. True for success, False otherwise.
        """
    try:
        _usergroup = self._zapi.usergroup.get({'output': 'extend', 'filter': {'name': [name]}})
        if len(_usergroup) > 0:
            return _usergroup
    except Exception as e:
        self._module.fail_json(msg="Failed to check if user group '%s' exists: %s" % (name, e))