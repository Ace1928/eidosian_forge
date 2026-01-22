from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def lookup_secret_non_msi(terms, vault_url, kwargs):
    client_id = kwargs['client_id'] if kwargs.get('client_id') else None
    secret = kwargs['secret'] if kwargs.get('secret') else None
    tenant_id = kwargs['tenant_id'] if kwargs.get('tenant_id') else None
    if all((v is not None for v in [client_id, secret, tenant_id])):
        credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=secret)
    else:
        credential = DefaultAzureCredential()
    client = SecretClient(vault_url, credential)
    ret = []
    for term in terms:
        try:
            secret_val = client.get_secret(term).value
            ret.append(secret_val)
        except Exception:
            raise AnsibleError('Failed to fetch secret ' + term + '.')
    return ret