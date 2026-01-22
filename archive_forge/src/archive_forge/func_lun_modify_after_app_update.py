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
def lun_modify_after_app_update(self, lun_path, results):
    if lun_path is None:
        lun_path = self.get_lun_path_from_backend(self.parameters['name'])
    current = self.get_lun(self.parameters['name'], lun_path)
    self.set_uuid(current)
    current.pop('name', None)
    lun_modify = self.na_helper.get_modified_attributes(current, self.parameters)
    if lun_modify:
        results['lun_modify_after_app_update'] = dict(lun_modify)
    self.check_for_errors(None, current, lun_modify)
    return lun_modify