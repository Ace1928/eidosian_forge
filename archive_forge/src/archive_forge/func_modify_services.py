from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def modify_services(self, modify, current):
    apis = {'fcp': 'protocols/san/fcp/services', 'iscsi': 'protocols/san/iscsi/services', 'nfs': 'protocols/nfs/services', 'nvme': 'protocols/nvme/services', 'ndmp': 'protocols/ndmp/svms'}
    for protocol, config in modify['services'].items():
        enabled = config.get('enabled')
        if enabled is None:
            continue
        api = apis.get(protocol)
        if not api:
            self.module.fail_json(msg='Internal error, unexpecting service: %s.' % protocol)
        if enabled:
            link = self.na_helper.safe_get(current, [protocol, '_links', 'self', 'href'])
        body = {'enabled': enabled}
        if enabled and (not link):
            body['svm.name'] = self.parameters['name']
            dummy, error = rest_generic.post_async(self.rest_api, api, body)
        else:
            dummy, error = rest_generic.patch_async(self.rest_api, api, current['uuid'], body)
        if error:
            self.module.fail_json(msg='Error in modify service for %s: %s' % (protocol, error))