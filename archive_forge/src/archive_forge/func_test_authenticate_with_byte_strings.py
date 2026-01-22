import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_authenticate_with_byte_strings(self):
    user = b'joe'
    unicode_pass = 'hìspass'
    utf8_pass = unicode_pass.encode('utf-8')
    factory = WideOpenSMTPFactory()
    conn = self.get_connection(b'[DEFAULT]\nsmtp_username=%s\nsmtp_password=%s\n' % (user, utf8_pass), smtp_factory=factory)
    self.assertEqual(unicode_pass, conn._smtp_password)
    conn._connect()
    self.assertEqual([('connect', 'localhost'), ('ehlo',), ('has_extn', 'starttls'), ('login', user, utf8_pass)], factory._calls)
    smtp_username, smtp_password = factory._calls[-1][1:]
    self.assertIsInstance(smtp_username, bytes)
    self.assertIsInstance(smtp_password, bytes)