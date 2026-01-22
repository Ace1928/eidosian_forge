import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_smtp_password_from_config(self):
    conn = self.get_connection(b'')
    self.assertIs(None, conn._smtp_password)
    conn = self.get_connection(b'smtp_password=mypass')
    self.assertEqual('mypass', conn._smtp_password)