from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def read_serverless_config(module):
    path = module.params.get('service_path')
    full_path = os.path.join(path, 'serverless.yml')
    try:
        with open(full_path) as sls_config:
            config = yaml.safe_load(sls_config.read())
            return config
    except IOError as e:
        module.fail_json(msg='Could not open serverless.yml in {0}. err: {1}'.format(full_path, str(e)))