from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible_collections.community.zabbix.plugins.module_utils.helpers import (
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_default_authentication(self):
    auth = self._zapi.authentication.get({'output': 'extend'})
    try:
        if auth['authentication_type'] == '0':
            return 'internal'
        elif auth['authentication_type'] == '1':
            return 'LDAP'
        else:
            self._module.fail_json(msg='Failed to query authentication type. Unknown authentication type %s' % auth)
    except Exception as e:
        self._module.fail_json(msg='Unhandled error while querying authentication type. %s' % e)