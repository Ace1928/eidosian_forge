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
def format_get_luns(self, records):
    luns = []
    if not records:
        return None
    for record in records:
        lun = {'uuid': self.na_helper.safe_get(record, ['uuid']), 'name': self.na_helper.safe_get(record, ['name']), 'path': self.na_helper.safe_get(record, ['name']), 'size': self.na_helper.safe_get(record, ['space', 'size']), 'comment': self.na_helper.safe_get(record, ['comment']), 'flexvol_name': self.na_helper.safe_get(record, ['location', 'volume', 'name']), 'os_type': self.na_helper.safe_get(record, ['os_type']), 'qos_policy_group': self.na_helper.safe_get(record, ['qos_policy', 'name']), 'space_reserve': self.na_helper.safe_get(record, ['space', 'guarantee', 'requested']), 'space_allocation': self.na_helper.safe_get(record, ['space', 'scsi_thin_provisioning_support_enabled'])}
        luns.append(lun)
    return luns