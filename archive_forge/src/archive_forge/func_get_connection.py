import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def get_connection(self, text, smtp_factory=None):
    my_config = config.MemoryStack(text)
    return smtp_connection.SMTPConnection(my_config, _smtp_factory=smtp_factory)