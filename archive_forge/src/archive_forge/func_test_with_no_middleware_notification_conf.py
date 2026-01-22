from unittest import mock
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_with_no_middleware_notification_conf(self):
    self.cfg.config(driver=['messaging'], group='oslo_messaging_notifications')
    self.cfg.config(driver=None, group='audit_middleware_notifications')
    app = self.create_simple_app()
    with mock.patch('oslo_messaging.notify.messaging.MessagingDriver.notify', side_effect=Exception('error')) as driver:
        app.get('/foo/bar', extra_environ=self.get_environ_header())
        self.assertTrue(driver.called)