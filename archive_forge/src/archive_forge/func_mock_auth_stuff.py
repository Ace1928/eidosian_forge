from __future__ import absolute_import, division, print_function
import pytest
from unittest import mock
from awx.main.models import User
@pytest.fixture
def mock_auth_stuff():
    """Some really specific session-related stuff is done for changing or setting
    passwords, so we will just avoid that here.
    """
    with mock.patch('awx.api.serializers.update_session_auth_hash'):
        yield