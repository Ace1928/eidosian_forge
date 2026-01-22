from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import \
def keycloak_clientsecret_module_resolve_params(module, kc):
    """
    Given an AnsibleModule definition for keycloak_clientsecret_*, and a
    KeycloakAPI client, resolve the params needed to interact with the Keycloak
    client secret, looking up the client by clientId if necessary via an API
    call.

    :return: tuple of id, realm
    """
    realm = module.params.get('realm')
    id = module.params.get('id')
    client_id = module.params.get('client_id')
    if id is None:
        client = kc.get_client_by_clientid(client_id, realm=realm)
        if client is None:
            module.fail_json(msg='Client does not exist {client_id}'.format(client_id=client_id))
        id = client['id']
    return (id, realm)