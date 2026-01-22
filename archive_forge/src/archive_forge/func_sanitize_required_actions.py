from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import KeycloakAPI, \
from ansible.module_utils.basic import AnsibleModule
def sanitize_required_actions(objects):
    for obj in objects:
        alias = obj['alias']
        name = obj['name']
        provider_id = obj['providerId']
        if not name:
            obj['name'] = alias
        if provider_id != alias:
            obj['providerId'] = alias
    return objects