from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def set_total_size(self, validate):
    attr = 'total_size'
    value = self.na_helper.safe_get(self.parameters, ['san_application_template', attr])
    if value is not None or not validate:
        self.parameters[attr] = value
        return
    lun_count = self.na_helper.safe_get(self.parameters, ['san_application_template', 'lun_count'])
    value = self.parameters.get('size')
    if value is not None and (lun_count is None or lun_count == 1):
        self.parameters[attr] = value
        return
    self.module.fail_json(msg="Error: 'total_size' is a required SAN application template attribute when creating a LUN application")