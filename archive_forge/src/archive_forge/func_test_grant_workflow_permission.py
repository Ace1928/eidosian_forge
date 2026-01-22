from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplate, User
@pytest.mark.django_db
@pytest.mark.parametrize('state', ('present', 'absent'))
def test_grant_workflow_permission(run_module, admin_user, organization, state):
    wfjt = WorkflowJobTemplate.objects.create(organization=organization, name='foo-workflow')
    rando = User.objects.create(username='rando')
    if state == 'absent':
        wfjt.execute_role.members.add(rando)
    result = run_module('role', {'user': rando.username, 'workflow': wfjt.name, 'role': 'execute', 'state': state}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    if state == 'present':
        assert rando in wfjt.execute_role
    else:
        assert rando not in wfjt.execute_role