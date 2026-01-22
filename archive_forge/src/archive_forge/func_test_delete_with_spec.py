from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplate, NotificationTemplate
@pytest.mark.django_db
def test_delete_with_spec(run_module, admin_user, organization, survey_spec):
    WorkflowJobTemplate.objects.create(organization=organization, name='foo-workflow', survey_enabled=True, survey_spec=survey_spec)
    result = run_module('workflow_job_template', {'name': 'foo-workflow', 'organization': organization.name, 'state': 'absent'}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', True), result
    assert WorkflowJobTemplate.objects.filter(name='foo-workflow', organization=organization).count() == 0