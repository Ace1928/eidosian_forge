from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplate, NotificationTemplate
@pytest.mark.django_db
def test_create_modify_no_survey(run_module, admin_user, organization, survey_spec):
    result = run_module('workflow_job_template', {'name': 'foo-workflow', 'organization': organization.name, 'job_tags': '', 'skip_tags': ''}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False), result
    wfjt = WorkflowJobTemplate.objects.get(name='foo-workflow')
    assert wfjt.organization_id == organization.id
    assert wfjt.survey_spec == {}
    result.pop('invocation', None)
    assert result == {'name': 'foo-workflow', 'id': wfjt.id, 'changed': True}
    result = run_module('workflow_job_template', {'name': 'foo-workflow', 'organization': organization.name}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert not result.get('changed', True), result