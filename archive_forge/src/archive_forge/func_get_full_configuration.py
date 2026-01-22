from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def get_full_configuration(self):
    try:
        rc, result = request(self.url + self.base_path, **self.creds)
        return result
    except Exception as err:
        self._logger.exception('Failed to retrieve the LDAP configuration.')
        self.module.fail_json(msg='Failed to retrieve LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))