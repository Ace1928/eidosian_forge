from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_associate_changed_status(run_module, admin_user, organization, project):
    jt = JobTemplate.objects.create(name='foo', project=project, playbook='helloworld.yml')
    labels = [Label.objects.create(name=f'foo{i}', organization=organization) for i in range(10)]
    result = run_module('job_template', dict(name=jt.name, playbook='helloworld.yml'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result['changed'] is False
    result = run_module('job_template', dict(name=jt.name, playbook='helloworld.yml', labels=[l.name for l in labels]), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result['changed']
    assert set((l.id for l in jt.labels.all())) == set((l.id for l in labels))
    random.shuffle(labels)
    result = run_module('job_template', dict(name=jt.name, playbook='helloworld.yml', labels=[l.name for l in labels]), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result['changed'] is False
    result = run_module('job_template', dict(name=jt.name, playbook='helloworld.yml'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result['changed'] is False
    fewer_labels = labels[:7]
    result = run_module('job_template', dict(name=jt.name, playbook='helloworld.yml', labels=[l.name for l in fewer_labels]), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result['changed']
    assert set((l.id for l in jt.labels.all())) == set((l.id for l in fewer_labels))