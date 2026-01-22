from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJobTemplateNode, WorkflowJobTemplate, JobTemplate, UnifiedJobTemplate
@pytest.fixture
def wfjt(organization):
    WorkflowJobTemplate.objects.create(organization=None, name='foo-workflow')
    return WorkflowJobTemplate.objects.create(organization=organization, name='foo-workflow')