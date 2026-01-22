from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Credential, CredentialType, Organization
@pytest.mark.django_db
def test_missing_credential_type(run_module, admin_user, organization):
    Organization.objects.create(name='test-org')
    result = run_module('credential', dict(name='A credential', organization=organization.name, credential_type='foobar', state='present'), admin_user)
    assert result.get('failed', False), result
    assert 'credential_type' in result['msg']
    assert 'foobar' in result['msg']
    assert 'returned 0 items, expected 1' in result['msg']