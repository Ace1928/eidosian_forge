from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import CredentialInputSource, Credential, CredentialType
@pytest.mark.django_db
def test_centrify_vault_credential_source(run_module, admin_user, organization, source_cred_centrify_secret, silence_deprecation):
    ct = CredentialType.defaults['ssh']()
    ct.save()
    tgt_cred = Credential.objects.create(name='Test Machine Credential', organization=organization, credential_type=ct, inputs={'username': 'bob'})
    result = run_module('credential_input_source', dict(source_credential=source_cred_centrify_secret.name, target_credential=tgt_cred.name, input_field_name='password', metadata={'system-name': 'systemname', 'account-name': 'accountname'}, state='present'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result
    assert CredentialInputSource.objects.count() == 1
    cis = CredentialInputSource.objects.first()
    assert cis.metadata['system-name'] == 'systemname'
    assert cis.metadata['account-name'] == 'accountname'
    assert cis.source_credential.name == source_cred_centrify_secret.name
    assert cis.target_credential.name == tgt_cred.name
    assert cis.input_field_name == 'password'
    assert result['id'] == cis.pk