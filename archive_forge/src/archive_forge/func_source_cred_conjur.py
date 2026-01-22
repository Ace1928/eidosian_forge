from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.fixture
def source_cred_conjur(organization):
    ct = CredentialType.defaults['conjur']()
    ct.save()
    return Credential.objects.create(name='CyberArk CONJUR Cred', credential_type=ct, inputs={'url': 'https://cyberark.example.com', 'api_key': 'myApiKey', 'account': 'account', 'username': 'username'})