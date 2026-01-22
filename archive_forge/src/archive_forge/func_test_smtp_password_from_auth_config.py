import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_smtp_password_from_auth_config(self):
    user = 'joe'
    password = 'hispass'
    factory = WideOpenSMTPFactory()
    conn = self.get_connection(b'[DEFAULT]\nsmtp_username=%s\n' % user.encode('ascii'), smtp_factory=factory)
    self.assertEqual(user, conn._smtp_username)
    self.assertIs(None, conn._smtp_password)
    conf = config.AuthenticationConfig()
    conf._get_config().update({'smtptest': {'scheme': 'smtp', 'user': user, 'password': password}})
    conf._save()
    conn._connect()
    self.assertEqual(password, conn._smtp_password)