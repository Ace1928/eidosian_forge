import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_create_connection_starttls(self):
    factory = StubSMTPFactory(smtp_features=['starttls'])
    conn = self.get_connection(b'', smtp_factory=factory)
    conn._create_connection()
    self.assertEqual([('connect', 'localhost'), ('ehlo',), ('has_extn', 'starttls'), ('starttls',), ('ehlo',)], factory._calls)