from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def test_multiple_lookup(run_module, admin_user):
    org1 = Organization.objects.create(name='foo')
    org2 = Organization.objects.create(name='bar')
    inv = Inventory.objects.create(name='Foo Inv')
    proj1 = Project.objects.create(name='foo', organization=org1, scm_type='git', scm_url='https://github.com/ansible/ansible-tower-samples')
    Project.objects.create(name='foo', organization=org2, scm_type='git', scm_url='https://github.com/ansible/ansible-tower-samples')
    result = run_module('job_template', {'name': 'Demo Job Template', 'project': proj1.name, 'inventory': inv.id, 'playbook': 'hello_world.yml'}, admin_user)
    assert result.get('failed', False)
    assert 'projects' in result['msg']
    assert 'foo' in result['msg']
    assert 'returned 2 items, expected 1' in result['msg']
    assert 'query' in result