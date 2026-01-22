from __future__ import (absolute_import, division, print_function)
import json
import time
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_current_catalog_settings(current_payload):
    catalog_payload = {}
    if current_payload.get('Filename') is not None:
        catalog_payload['Filename'] = current_payload['Filename']
    if current_payload.get('SourcePath') is not None:
        catalog_payload['SourcePath'] = current_payload['SourcePath']
    repository_dict = {'Name': current_payload['Repository'].get('Name'), 'Id': current_payload['Repository'].get('Id'), 'Description': current_payload['Repository'].get('Description'), 'RepositoryType': current_payload['Repository'].get('RepositoryType'), 'Source': current_payload['Repository'].get('Source'), 'DomainName': current_payload['Repository'].get('DomainName'), 'Username': current_payload['Repository'].get('Username'), 'Password': current_payload['Repository'].get('Password'), 'CheckCertificate': current_payload['Repository'].get('CheckCertificate')}
    repository_payload = dict([(k, v) for k, v in repository_dict.items() if v is not None])
    if repository_payload:
        catalog_payload['Repository'] = repository_payload
    return catalog_payload