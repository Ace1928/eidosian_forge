from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_usrgrpid_from_name(self, usrgrp):
    usrgrpids = self._zapi.usergroup.get({'filter': {'name': usrgrp}})
    if not usrgrpids or len(usrgrpids) > 1:
        self._module.fail_json("User group '%s' cannot be found" % usrgrp)
    return usrgrpids[0]['usrgrpid']