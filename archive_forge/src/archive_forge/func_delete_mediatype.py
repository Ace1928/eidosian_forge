from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def delete_mediatype(self, mediatype_id):
    try:
        return self._zapi.mediatype.delete([mediatype_id])
    except Exception as e:
        self._module.fail_json(msg="Failed to delete mediatype '{_id}': {e}".format(_id=mediatype_id, e=e))