from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_job_template_with_wrong_survey_spec(run_module, admin_user, project, inventory, survey_spec):
    result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, inventory=inventory.name, survey_spec=survey_spec, survey_enabled=True), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False), result
    jt = JobTemplate.objects.get(pk=result['id'])
    assert jt.survey_spec == survey_spec
    prior_ct = ActivityStream.objects.count()
    del survey_spec['description']
    result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, inventory=inventory.name, survey_spec=survey_spec, survey_enabled=True), admin_user)
    assert result.get('failed', True)
    assert result.get('msg') == "Failed to update survey: Field 'description' is missing from survey spec."
    assert ActivityStream.objects.count() == prior_ct