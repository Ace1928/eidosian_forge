import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_create_connection_ehlo_fails(self):
    factory = StubSMTPFactory(fail_on=['ehlo'])
    conn = self.get_connection(b'', smtp_factory=factory)
    conn._create_connection()
    self.assertEqual([('connect', 'localhost'), ('ehlo',), ('helo',), ('has_extn', 'starttls')], factory._calls)