from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, Group, Host
@pytest.mark.django_db
def test_associate_hosts_and_children(run_module, admin_user, organization):
    inv = Inventory.objects.create(name='test-inv', organization=organization)
    group = Group.objects.create(name='Test Group', inventory=inv)
    inv_hosts = [Host.objects.create(inventory=inv, name='foo{0}'.format(i)) for i in range(3)]
    group.hosts.add(inv_hosts[0], inv_hosts[1])
    child = Group.objects.create(inventory=inv, name='child_group')
    result = run_module('group', dict(name='Test Group', inventory='test-inv', hosts=[inv_hosts[1].name, inv_hosts[2].name], children=[child.name], state='present'), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result['changed'] is True
    assert set(group.hosts.all()) == set([inv_hosts[1], inv_hosts[2]])
    assert set(group.children.all()) == set([child])