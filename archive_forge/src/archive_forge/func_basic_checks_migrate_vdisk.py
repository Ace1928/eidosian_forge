from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def basic_checks_migrate_vdisk(self):
    self.log('Entering function basic_checks_migrate_vdisk()')
    invalid_params = {}
    missing = [item[0] for item in [('new_pool', self.new_pool), ('source_volume', self.source_volume)] if not item[1]]
    if missing:
        self.module.fail_json(msg='Missing mandatory parameter: [{0}] for migration across pools'.format(', '.join(missing)))
    invalid_params['across_pools'] = ['state', 'relationship_name', 'remote_cluster', 'remote_username', 'remote_password', 'remote_token', 'remote_pool', 'remote_validate_certs', 'replicate_hosts']
    param_list = set(invalid_params['across_pools'])
    for param in param_list:
        if self.type_of_migration == 'across_pools':
            if getattr(self, param):
                if param in invalid_params['across_pools']:
                    self.module.fail_json(msg="Invalid parameter [%s] for volume migration 'across_pools'" % param)