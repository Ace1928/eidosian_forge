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
def setup_rest_application(self):
    use_application_template = self.na_helper.safe_get(self.parameters, ['san_application_template', 'use_san_application'])
    rest_app = None
    if self.use_rest:
        if use_application_template:
            if self.parameters.get('flexvol_name') is not None:
                self.module.fail_json(msg="'flexvol_name' option is not supported when san_application_template is present")
            if self.parameters.get('qtree_name') is not None:
                self.module.fail_json(msg="'qtree_name' option is not supported when san_application_template is present")
            name = self.na_helper.safe_get(self.parameters, ['san_application_template', 'name'], allow_sparse_dict=False)
            rest_app = RestApplication(self.rest_api, self.parameters['vserver'], name)
        elif self.parameters.get('flexvol_name') is None:
            self.module.fail_json(msg='flexvol_name option is required when san_application_template is not present')
    else:
        if use_application_template:
            self.module.fail_json(msg='Error: using san_application_template requires ONTAP 9.7 or later and REST must be enabled.')
        if self.parameters.get('flexvol_name') is None:
            self.module.fail_json(msg="Error: 'flexvol_name' option is required when using ZAPI.")
    return rest_app