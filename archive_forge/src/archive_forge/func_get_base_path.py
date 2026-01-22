from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def get_base_path(self):
    embedded = self.is_embedded()
    if embedded:
        return 'storage-systems/%s/ldap/' % self.ssid
    else:
        return '/ldap/'