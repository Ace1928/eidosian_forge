from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_user_by_user_name(self, user_name):
    """Get user by user name

        Args:
            user_name: user name

        Returns:
            user matching user name

        """
    try:
        filter = {'username': [user_name]}
        user_list = self._zapi.user.get({'output': 'extend', 'filter': filter})
        if len(user_list) < 1:
            self._module.fail_json(msg='User not found: %s' % user_name)
        else:
            return user_list[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get user '%s': %s" % (user_name, e))