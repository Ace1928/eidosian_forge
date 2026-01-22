from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def keycloak_argument_spec():
    """
    Returns argument_spec of options common to keycloak_*-modules

    :return: argument_spec dict
    """
    return dict(auth_keycloak_url=dict(type='str', aliases=['url'], required=True, no_log=False), auth_client_id=dict(type='str', default='admin-cli'), auth_realm=dict(type='str'), auth_client_secret=dict(type='str', default=None, no_log=True), auth_username=dict(type='str', aliases=['username']), auth_password=dict(type='str', aliases=['password'], no_log=True), validate_certs=dict(type='bool', default=True), connection_timeout=dict(type='int', default=10), token=dict(type='str', no_log=True), http_agent=dict(type='str', default='Ansible'))