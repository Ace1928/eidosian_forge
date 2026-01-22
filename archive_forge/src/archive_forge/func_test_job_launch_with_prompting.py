from __future__ import absolute_import, division, print_function
import random
import pytest
from awx.main.models import ActivityStream, JobTemplate, Job, NotificationTemplate, Label
@pytest.mark.django_db
def test_job_launch_with_prompting(run_module, admin_user, project, organization, inventory, machine_credential):
    JobTemplate.objects.create(name='foo', project=project, organization=organization, playbook='helloworld.yml', ask_variables_on_launch=True, ask_inventory_on_launch=True, ask_credential_on_launch=True)
    result = run_module('job_launch', dict(job_template='foo', inventory=inventory.name, credential=machine_credential.name, extra_vars={'var1': 'My First Variable', 'var2': 'My Second Variable', 'var3': 'My Third Variable'}), admin_user)
    assert result.pop('changed', None), result
    job = Job.objects.get(id=result['id'])
    assert job.extra_vars == '{"var1": "My First Variable", "var2": "My Second Variable", "var3": "My Third Variable"}'
    assert job.inventory == inventory
    assert [cred.id for cred in job.credentials.all()] == [machine_credential.id]