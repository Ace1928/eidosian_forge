from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJob
@pytest.mark.django_db
def test_bulk_job_launch(run_module, admin_user, job_template):
    jobs = [dict(unified_job_template=job_template.id)]
    result = run_module('bulk_job_launch', {'name': 'foo-bulk-job', 'jobs': jobs, 'extra_vars': {'animal': 'owl'}, 'limit': 'foo', 'wait': False}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result
    bulk_job = WorkflowJob.objects.get(name='foo-bulk-job')
    assert bulk_job.extra_vars == '{"animal": "owl"}'
    assert bulk_job.limit == 'foo'