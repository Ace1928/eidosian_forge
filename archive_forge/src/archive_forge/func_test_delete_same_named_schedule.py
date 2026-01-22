from __future__ import absolute_import, division, print_function
import pytest
from ansible.errors import AnsibleError
from awx.main.models import JobTemplate, Schedule
from awx.api.serializers import SchedulePreviewSerializer
@pytest.mark.django_db
def test_delete_same_named_schedule(run_module, project, inventory, admin_user):
    jt1 = JobTemplate.objects.create(name='jt1', project=project, inventory=inventory, playbook='helloworld.yml')
    jt2 = JobTemplate.objects.create(name='jt2', project=project, inventory=inventory, playbook='helloworld2.yml')
    Schedule.objects.create(name='Some Schedule', rrule='DTSTART:20300112T210000Z RRULE:FREQ=DAILY;INTERVAL=1', unified_job_template=jt1)
    Schedule.objects.create(name='Some Schedule', rrule='DTSTART:20300112T210000Z RRULE:FREQ=DAILY;INTERVAL=1', unified_job_template=jt2)
    result = run_module('schedule', {'name': 'Some Schedule', 'unified_job_template': 'jt1', 'state': 'absent'}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert Schedule.objects.filter(name='Some Schedule').count() == 1