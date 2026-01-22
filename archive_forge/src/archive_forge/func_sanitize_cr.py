from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak import KeycloakAPI, camel, \
from ansible.module_utils.basic import AnsibleModule
def sanitize_cr(realmrep):
    """ Removes probably sensitive details from a realm representation.

    :param realmrep: the realmrep dict to be sanitized
    :return: sanitized realmrep dict
    """
    result = realmrep.copy()
    if 'secret' in result:
        result['secret'] = '********'
    if 'attributes' in result:
        if 'saml.signing.private.key' in result['attributes']:
            result['attributes'] = result['attributes'].copy()
            result['attributes']['saml.signing.private.key'] = '********'
    return result