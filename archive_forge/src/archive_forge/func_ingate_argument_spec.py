from __future__ import (absolute_import, division, print_function)
def ingate_argument_spec(**kwargs):
    client_options = dict(version=dict(choices=['v1'], default='v1'), scheme=dict(choices=['http', 'https'], required=True), address=dict(type='str', required=True), username=dict(type='str', required=True), password=dict(type='str', required=True, no_log=True), port=dict(type='int'), timeout=dict(type='int'), validate_certs=dict(default=True, type='bool', aliases=['verify_ssl']))
    argument_spec = dict(client=dict(type='dict', required=True, options=client_options))
    argument_spec.update(kwargs)
    return argument_spec