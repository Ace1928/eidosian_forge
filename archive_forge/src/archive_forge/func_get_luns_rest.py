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
def get_luns_rest(self, lun_path=None):
    if lun_path is None and self.parameters.get('flexvol_name') is None:
        return []
    api = 'storage/luns'
    query = {'svm.name': self.parameters['vserver'], 'fields': 'comment,lun_maps,name,os_type,qos_policy.name,space'}
    if lun_path is not None:
        query['name'] = lun_path
    else:
        query['location.volume.name'] = self.parameters['flexvol_name']
        if self.parameters.get('qtree_name') is not None:
            query['location.qtree.name'] = self.parameters['qtree_name']
    record, error = rest_generic.get_0_or_more_records(self.rest_api, api, query)
    if error:
        if lun_path is not None:
            self.module.fail_json(msg='Error getting lun_path %s: %s' % (lun_path, to_native(error)), exception=traceback.format_exc())
        else:
            self.module.fail_json(msg="Error getting LUN's for flexvol %s: %s" % (self.parameters['flexvol_name'], to_native(error)), exception=traceback.format_exc())
    return self.format_get_luns(record)