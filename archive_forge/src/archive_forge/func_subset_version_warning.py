from __future__ import absolute_import, division, print_function
import codecs
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_owning_resource, rest_vserver
def subset_version_warning(self, get_ontap_subset_info):
    unsupported_subset = []
    warn_message = ''
    user_version = self.rest_api.get_ontap_version()
    for subset in self.parameters['gather_subset']:
        if subset in get_ontap_subset_info and 'version' in get_ontap_subset_info[subset] and (get_ontap_subset_info[subset]['version'] > user_version):
            warn_message += '%s requires %s, ' % (subset, get_ontap_subset_info[subset]['version'])
            unsupported_subset.append(subset)
            self.parameters['gather_subset'].remove(subset)
    if warn_message != '':
        self.module.warn('The following subset have been removed from your query as they are not supported on your version of ONTAP %s' % warn_message)
    return unsupported_subset