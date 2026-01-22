from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import (
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import \
def keycloak_clientscope_type_module():
    """
    Returns an AnsibleModule definition.

    :return: argument_spec dict
    """
    argument_spec = keycloak_argument_spec()
    meta_args = dict(realm=dict(default='master'), client_id=dict(type='str', aliases=['clientId']), default_clientscopes=dict(type='list', elements='str'), optional_clientscopes=dict(type='list', elements='str'))
    argument_spec.update(meta_args)
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True, required_one_of=[['token', 'auth_realm', 'auth_username', 'auth_password'], ['default_clientscopes', 'optional_clientscopes']], required_together=[['auth_realm', 'auth_username', 'auth_password']], mutually_exclusive=[['token', 'auth_realm'], ['token', 'auth_username'], ['token', 'auth_password']])
    return module