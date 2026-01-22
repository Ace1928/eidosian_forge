from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_if_drule_exists(self, name):
    """Check if discovery rule exists.
        Args:
            name: Name of the discovery rule.
        Returns:
            The return value. True for success, False otherwise.
        """
    try:
        _drule = self._zapi.drule.get({'output': 'extend', 'selectDChecks': 'extend', 'filter': {'name': [name]}})
        if len(_drule) > 0:
            return _drule
    except Exception as e:
        self._module.fail_json(msg="Failed to check if discovery rule '%s' exists: %s" % (name, e))