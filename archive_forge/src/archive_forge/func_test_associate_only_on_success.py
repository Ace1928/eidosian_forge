from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_associate_only_on_success(run_module, admin_user, organization, project):
    jt = JobTemplate.objects.create(name='foo', project=project, playbook='helloworld.yml', ask_inventory_on_launch=True)
    create_kwargs = dict(notification_configuration={'url': 'http://www.example.com/hook', 'headers': {'X-Custom-Header': 'value123'}, 'password': 'bar'}, notification_type='webhook', organization=organization)
    nt1 = NotificationTemplate.objects.create(name='nt1', **create_kwargs)
    nt2 = NotificationTemplate.objects.create(name='nt2', **create_kwargs)
    jt.notification_templates_error.add(nt1)
    result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, notification_templates_success=['nt2']), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', True), result
    assert list(jt.notification_templates_success.values_list('id', flat=True)) == [nt2.id]
    assert list(jt.notification_templates_error.values_list('id', flat=True)) == [nt1.id]
    result = run_module('job_template', dict(name='foo', playbook='helloworld.yml', project=project.name, notification_templates_success=[]), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', True), result
    assert list(jt.notification_templates_success.values_list('id', flat=True)) == []
    assert list(jt.notification_templates_error.values_list('id', flat=True)) == [nt1.id]