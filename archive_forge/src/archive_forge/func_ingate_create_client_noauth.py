from __future__ import (absolute_import, division, print_function)
def ingate_create_client_noauth(**kwargs):
    client_params = kwargs['client']
    api_client = ingatesdk.Client(client_params['version'], client_params['scheme'], client_params['address'], client_params['username'], client_params['password'], port=client_params['port'], timeout=client_params['timeout'])
    verify_ssl = client_params.get('validate_certs')
    if not verify_ssl:
        api_client.skip_verify_certificate()
    return api_client