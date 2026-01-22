import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_smtp_username(self):
    conn = self.get_connection(b'')
    self.assertIs(None, conn._smtp_username)
    conn = self.get_connection(b'smtp_username=joebody')
    self.assertEqual('joebody', conn._smtp_username)