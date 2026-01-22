import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_destination_address_required(self):
    msg = Message()
    msg['From'] = '"J. Random Developer" <jrandom@example.com>'
    self.assertRaises(smtp_connection.NoDestinationAddress, smtp_connection.SMTPConnection(config.MemoryStack(b'')).send_email, msg)
    msg = email_message.EmailMessage('from@from.com', '', 'subject')
    self.assertRaises(smtp_connection.NoDestinationAddress, smtp_connection.SMTPConnection(config.MemoryStack(b'')).send_email, msg)
    msg = email_message.EmailMessage('from@from.com', [], 'subject')
    self.assertRaises(smtp_connection.NoDestinationAddress, smtp_connection.SMTPConnection(config.MemoryStack(b'')).send_email, msg)