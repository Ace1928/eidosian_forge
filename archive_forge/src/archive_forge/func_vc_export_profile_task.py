from __future__ import absolute_import, division, print_function
import_profile:
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
import json
import time
def vc_export_profile_task(self):
    profile_spec = self.get_profile_spec()
    infra = self.api_client.appliance.infraprofile.Configs
    config_json = infra.export(spec=profile_spec)
    if self.config_path is None:
        self.config_path = self.params.get('api') + '.json'
    parsed = json.loads(config_json)
    with open(self.config_path, 'w', encoding='utf-8') as outfile:
        json.dump(parsed, outfile, ensure_ascii=False, indent=2)
    self.module.exit_json(changed=False, export_config_json=config_json)