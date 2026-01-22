from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import \
def keycloak_clientsecret_module():
    """
    Returns an AnsibleModule definition for modules that interact with a client
    secret.

    :return: argument_spec dict
    """
    argument_spec = keycloak_argument_spec()
    meta_args = dict(realm=dict(default='master'), id=dict(type='str'), client_id=dict(type='str', aliases=['clientId']))
    argument_spec.update(meta_args)
    module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True, required_one_of=[['id', 'client_id'], ['token', 'auth_realm', 'auth_username', 'auth_password']], required_together=[['auth_realm', 'auth_username', 'auth_password']], mutually_exclusive=[['token', 'auth_realm'], ['token', 'auth_username'], ['token', 'auth_password']])
    return module