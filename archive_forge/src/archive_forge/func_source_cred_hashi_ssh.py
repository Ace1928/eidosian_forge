from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.fixture
def source_cred_hashi_ssh(organization):
    ct = CredentialType.defaults['hashivault_ssh']()
    ct.save()
    return Credential.objects.create(name='HashiCorp ssh Cred', credential_type=ct, inputs={'url': 'https://ssh.hash.example.com', 'token': 'myApiKey', 'role_id': 'role', 'secret_id': 'secret'})