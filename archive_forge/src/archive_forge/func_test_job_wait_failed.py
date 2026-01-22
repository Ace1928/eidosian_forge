from __future__ import absolute_import, division, print_function
import pytest
from django.utils.timezone import now
from awx.main.models import Job
@pytest.mark.django_db
def test_job_wait_failed(run_module, admin_user):
    job = Job.objects.create(status='failed', started=now(), finished=now())
    result = run_module('job_wait', dict(job_id=job.id), admin_user)
    result.pop('invocation', None)
    result['elapsed'] = float(result['elapsed'])
    assert result.pop('finished', '')[:10] == str(job.finished)[:10]
    assert result.pop('started', '')[:10] == str(job.started)[:10]
    assert result == {'status': 'failed', 'failed': True, 'changed': False, 'elapsed': job.elapsed, 'id': job.id, 'msg': 'Job with id 1 failed'}