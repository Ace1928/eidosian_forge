from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import WorkflowJob
@pytest.mark.django_db
def test_bulk_host_delete(run_module, admin_user, inventory):
    hosts = [dict(name='127.0.0.1'), dict(name='foo.dns.org')]
    result = run_module('bulk_host_create', {'inventory': inventory.name, 'hosts': hosts}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result
    resp_hosts_ids = list(inventory.hosts.all().values_list('id', flat=True))
    result = run_module('bulk_host_delete', {'hosts': resp_hosts_ids}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result