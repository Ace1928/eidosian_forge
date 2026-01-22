from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
Checks if value map exists.

        Args:
            name: Zabbix valuemap name

        Returns:
            tuple: First element is True if valuemap exists and False otherwise.
                Second element is a dictionary of valuemap object if it exists.
        