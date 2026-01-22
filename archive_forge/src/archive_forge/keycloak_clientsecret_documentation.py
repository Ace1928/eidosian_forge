from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import \

    Given an AnsibleModule definition for keycloak_clientsecret_*, and a
    KeycloakAPI client, resolve the params needed to interact with the Keycloak
    client secret, looking up the client by clientId if necessary via an API
    call.

    :return: tuple of id, realm
    