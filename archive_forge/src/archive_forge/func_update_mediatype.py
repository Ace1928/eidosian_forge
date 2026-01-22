from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_mediatype(self, **kwargs):
    try:
        self._zapi.mediatype.update(kwargs)
    except Exception as e:
        self._module.fail_json(msg="Failed to update mediatype '{_id}': {e}".format(_id=kwargs['mediatypeid'], e=e))