import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_missing_server(self):
    conn = self.get_connection(b'', smtp_factory=connection_refuser)
    self.assertRaises(smtp_connection.DefaultSMTPConnectionRefused, conn._connect)
    conn = self.get_connection(b'smtp_server=smtp.example.com', smtp_factory=connection_refuser)
    self.assertRaises(smtp_connection.SMTPConnectionRefused, conn._connect)