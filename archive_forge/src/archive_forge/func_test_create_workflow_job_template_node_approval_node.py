from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplateNode, WorkflowJobTemplate, JobTemplate, UnifiedJobTemplate
@pytest.mark.django_db
def test_create_workflow_job_template_node_approval_node(run_module, admin_user, wfjt, job_template):
    """This is a part of the API contract for creating approval nodes"""
    this_identifier = '42🐉'
    result = run_module('workflow_job_template_node', {'identifier': this_identifier, 'workflow_job_template': wfjt.name, 'organization': wfjt.organization.name, 'approval_node': {'name': 'foo-jt-approval'}}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', False), result
    node = WorkflowJobTemplateNode.objects.get(identifier=this_identifier)
    approval_node = UnifiedJobTemplate.objects.get(name='foo-jt-approval')
    assert result['id'] == approval_node.id
    assert node.identifier == this_identifier
    assert node.workflow_job_template_id == wfjt.id
    assert node.unified_job_template_id is approval_node.id