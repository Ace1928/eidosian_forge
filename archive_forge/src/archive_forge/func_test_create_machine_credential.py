from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Credential, CredentialType, Organization
@pytest.mark.django_db
def test_create_machine_credential(run_module, admin_user, organization):
    Organization.objects.create(name='test-org')
    ct = CredentialType.defaults['ssh']()
    ct.save()
    result = run_module('credential', dict(name='Test Machine Credential', organization=organization.name, credential_type='Machine', state='present'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result
    cred = Credential.objects.get(name='Test Machine Credential')
    assert cred.credential_type == ct
    assert result['name'] == 'Test Machine Credential'
    assert result['id'] == cred.pk