from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.fixture
def source_cred_centrify_secret(organization):
    ct = CredentialType.defaults['centrify_vault_kv']()
    ct.save()
    return Credential.objects.create(name='Centrify vault secret Cred', credential_type=ct, inputs={'url': 'https://tenant_id.my.centrify-dev.net', 'client_id': 'secretuser@tenant', 'client_password': 'secretuserpassword'})