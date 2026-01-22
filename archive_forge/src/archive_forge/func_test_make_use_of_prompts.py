from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplateNode, WorkflowJobTemplate, JobTemplate, UnifiedJobTemplate
@pytest.mark.django_db
def test_make_use_of_prompts(run_module, admin_user, wfjt, job_template, machine_credential, vault_credential):
    result = run_module('workflow_job_template_node', {'identifier': '42', 'workflow_job_template': 'foo-workflow', 'organization': wfjt.organization.name, 'unified_job_template': 'foo-jt', 'extra_data': {'foo': 'bar', 'another-foo': {'barz': 'bar2'}}, 'limit': 'foo_hosts', 'credentials': [machine_credential.name, vault_credential.name], 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False)
    node = WorkflowJobTemplateNode.objects.get(identifier='42')
    assert node.limit == 'foo_hosts'
    assert node.extra_data == {'foo': 'bar', 'another-foo': {'barz': 'bar2'}}
    assert set(node.credentials.all()) == set([machine_credential, vault_credential])