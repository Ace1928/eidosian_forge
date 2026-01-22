from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplateNode, WorkflowJobTemplate, JobTemplate, UnifiedJobTemplate
@pytest.mark.django_db
def test_create_workflow_job_template_node(run_module, admin_user, wfjt, job_template):
    this_identifier = '42🐉'
    result = run_module('workflow_job_template_node', {'identifier': this_identifier, 'workflow_job_template': 'foo-workflow', 'organization': wfjt.organization.name, 'unified_job_template': 'foo-jt', 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    node = WorkflowJobTemplateNode.objects.get(identifier=this_identifier)
    result.pop('invocation', None)
    assert result == {'name': this_identifier, 'id': node.id, 'changed': True}
    assert node.identifier == this_identifier
    assert node.workflow_job_template_id == wfjt.id
    assert node.unified_job_template_id == job_template.id