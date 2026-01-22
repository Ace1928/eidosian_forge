import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def starttls(self):
    self._calls.append(('starttls',))
    if 'starttls' in self._fail_on:
        return (500, 'starttls failure')
    else:
        self._ehlo_called = True
        return (200, 'starttls success')