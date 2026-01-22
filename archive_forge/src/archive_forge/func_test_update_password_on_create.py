from __future__ import absolute_import, division, print_function
import pytest
from unittest import mock
from awx.main.models import User
@pytest.mark.django_db
def test_update_password_on_create(run_module, admin_user, mock_auth_stuff):
    for i in range(2):
        result = run_module('user', dict(username='Bob', password='pass4word', update_secrets=False), admin_user)
        assert not result.get('failed', False), result.get('msg', result)
    assert not result.get('changed')