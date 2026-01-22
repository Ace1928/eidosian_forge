from __future__ import absolute_import, division, print_function
import pytest
from django.utils.timezone import now
from awx.main.models import Job
@pytest.mark.django_db
def test_job_wait_not_found(run_module, admin_user):
    result = run_module('job_wait', dict(job_id=42), admin_user)
    result.pop('invocation', None)
    assert result == {'failed': True, 'msg': 'Unable to wait on job 42; that ID does not exist.'}