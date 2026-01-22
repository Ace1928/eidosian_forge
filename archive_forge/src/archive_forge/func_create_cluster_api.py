from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def create_cluster_api(self, options):
    """ Call send_request directly rather than using the SDK if new fields are present
            The new SDK will support these in version 1.17 (Nov or Feb)
        """
    extra_options = ['enableSoftwareEncryptionAtRest', 'orderNumber', 'serialNumber']
    if not any((item in options for item in extra_options)):
        return self.sfe_cluster.create_cluster(**options)
    params = {'mvip': options['mvip'], 'svip': options['svip'], 'repCount': options['rep_count'], 'username': options['username'], 'password': options['password'], 'nodes': options['nodes']}
    if options['accept_eula'] is not None:
        params['acceptEula'] = options['accept_eula']
    if options['attributes'] is not None:
        params['attributes'] = options['attributes']
    for option in extra_options:
        if options.get(option):
            params[option] = options[option]
    return self.sfe_cluster.send_request('CreateCluster', netapp_utils.solidfire.CreateClusterResult, params, since=None)