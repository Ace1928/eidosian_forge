from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_rest_create_body(self, mappings):
    """ maps self.parameters to REST API body attributes, using mappings to identify fields to add """
    body = {}
    for key, value in mappings.items():
        if key in self.parameters:
            if key == 'aggr_list':
                body[value] = [dict(name=aggr) for aggr in self.parameters[key]]
            else:
                body[value] = self.parameters[key]
        elif key == 'origins':
            origin = dict(volume=dict(name=self.parameters['origin_volume']), svm=dict(name=self.parameters['origin_vserver']))
            body[value] = [origin]
    return body