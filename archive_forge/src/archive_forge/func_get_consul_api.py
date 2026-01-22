from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
def get_consul_api(module):
    return consul.Consul(host=module.params.get('host'), port=module.params.get('port'), scheme=module.params.get('scheme'), verify=module.params.get('validate_certs'), token=module.params.get('token'))