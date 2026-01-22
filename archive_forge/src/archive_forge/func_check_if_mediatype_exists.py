from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_if_mediatype_exists(self, name):
    """Checks if mediatype exists.

        Args:
            name: Zabbix mediatype name

        Returns:
            Tuple of (True, `id of the mediatype`) if mediatype exists, (False, None) otherwise
        """
    filter_key_name = 'description'
    filter_key_name = 'name'
    try:
        mediatype_list = self._zapi.mediatype.get({'output': 'extend', 'filter': {filter_key_name: [name]}})
        if len(mediatype_list) < 1:
            return (False, None)
        else:
            return (True, mediatype_list[0]['mediatypeid'])
    except Exception as e:
        self._module.fail_json(msg="Failed to get ID of the mediatype '{name}': {e}".format(name=name, e=e))