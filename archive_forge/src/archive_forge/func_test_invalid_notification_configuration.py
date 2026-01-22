from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import NotificationTemplate, Job
@pytest.mark.django_db
def test_invalid_notification_configuration(run_module, admin_user, organization):
    result = run_module('notification_template', dict(name='foo-notification-template', organization=organization.name, notification_type='email', notification_configuration={}), admin_user)
    assert result.get('failed', False), result.get('msg', result)
    assert 'Missing required fields for Notification Configuration' in result['msg']