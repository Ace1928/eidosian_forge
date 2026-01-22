from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_storage_provider(self, type='primary'):
    args = {'type': type}
    provider = self.module.params.get('provider')
    storage_providers = self.query_api('listStorageProviders', **args)
    for sp in storage_providers.get('dataStoreProvider') or []:
        if sp['name'].lower() == provider.lower():
            return provider
    self.fail_json(msg='Storage provider %s not found' % provider)