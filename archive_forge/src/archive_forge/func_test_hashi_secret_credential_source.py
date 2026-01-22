from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.mark.django_db
def test_hashi_secret_credential_source(run_module, admin_user, organization, source_cred_hashi_secret, silence_deprecation):
    ct = CredentialType.defaults['ssh']()
    ct.save()
    tgt_cred = Credential.objects.create(name='Test Machine Credential', organization=organization, credential_type=ct, inputs={'username': 'bob'})
    result = run_module('credential_input_source', dict(source_credential=source_cred_hashi_secret.name, target_credential=tgt_cred.name, input_field_name='password', metadata={'secret_path': '/path/to/secret', 'auth_path': '/path/to/auth', 'secret_backend': 'backend', 'secret_key': 'a_key'}, state='present'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result
    assert CredentialInputSource.objects.count() == 1
    cis = CredentialInputSource.objects.first()
    assert cis.metadata['secret_path'] == '/path/to/secret'
    assert cis.metadata['auth_path'] == '/path/to/auth'
    assert cis.metadata['secret_backend'] == 'backend'
    assert cis.metadata['secret_key'] == 'a_key'
    assert cis.source_credential.name == source_cred_hashi_secret.name
    assert cis.target_credential.name == tgt_cred.name
    assert cis.input_field_name == 'password'
    assert result['id'] == cis.pk