from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Inventory
@pytest.mark.django_db
def test_invalid_smart_inventory_create(run_module, admin_user, organization):
    result = run_module('inventory', {'name': 'foo-inventory', 'organization': organization.name, 'kind': 'smart', 'host_filter': 'ansible', 'state': 'present'}, admin_user)
    assert result.get('failed', False), result
    assert 'Invalid query ansible' in result['msg']