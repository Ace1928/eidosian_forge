from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplate, User
@pytest.mark.django_db
def test_invalid_role(run_module, admin_user, project):
    rando = User.objects.create(username='rando')
    result = run_module('role', {'user': rando.username, 'project': project.name, 'role': 'adhoc', 'state': 'present'}, admin_user)
    assert result.get('failed', False)
    msg = result.get('msg')
    assert 'has no role adhoc_role' in msg
    assert 'available roles: admin_role, use_role, update_role, read_role' in msg