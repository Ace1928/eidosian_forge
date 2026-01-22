from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, Group, Host
@pytest.mark.django_db
def test_group_idempotent(run_module, admin_user):
    org = Organization.objects.create(name='test-org')
    inv = Inventory.objects.create(name='test-inv', organization=org)
    group = Group.objects.create(name='Test Group', inventory=inv)
    result = run_module('group', dict(name='Test Group', inventory='test-inv', state='present'), admin_user)
    result.pop('invocation')
    assert result == {'id': group.id, 'changed': False}