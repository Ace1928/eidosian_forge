from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
import_nomad = None
def transform_response(nomad_response):
    transformed_response = {'accessor_id': nomad_response['AccessorID'], 'create_index': nomad_response['CreateIndex'], 'create_time': nomad_response['CreateTime'], 'expiration_ttl': nomad_response['ExpirationTTL'], 'expiration_time': nomad_response['ExpirationTime'], 'global': nomad_response['Global'], 'hash': nomad_response['Hash'], 'modify_index': nomad_response['ModifyIndex'], 'name': nomad_response['Name'], 'policies': nomad_response['Policies'], 'roles': nomad_response['Roles'], 'secret_id': nomad_response['SecretID'], 'type': nomad_response['Type']}
    return transformed_response