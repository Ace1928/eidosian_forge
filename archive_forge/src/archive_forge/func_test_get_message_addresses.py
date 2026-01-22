import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_get_message_addresses(self):
    msg = Message()
    from_, to = smtp_connection.SMTPConnection.get_message_addresses(msg)
    self.assertEqual('', from_)
    self.assertEqual([], to)
    msg['From'] = '"J. Random Developer" <jrandom@example.com>'
    msg['To'] = 'John Doe <john@doe.com>, Jane Doe <jane@doe.com>'
    msg['CC'] = 'Pepe Pérez <pperez@ejemplo.com>'
    msg['Bcc'] = 'user@localhost'
    from_, to = smtp_connection.SMTPConnection.get_message_addresses(msg)
    self.assertEqual('jrandom@example.com', from_)
    self.assertEqual(sorted(['john@doe.com', 'jane@doe.com', 'pperez@ejemplo.com', 'user@localhost']), sorted(to))
    msg = email_message.EmailMessage('"J. Random Developer" <jrandom@example.com>', ['John Doe <john@doe.com>', 'Jane Doe <jane@doe.com>', 'Pepe Pérez <pperez@ejemplo.com>', 'user@localhost'], 'subject')
    from_, to = smtp_connection.SMTPConnection.get_message_addresses(msg)
    self.assertEqual('jrandom@example.com', from_)
    self.assertEqual(sorted(['john@doe.com', 'jane@doe.com', 'pperez@ejemplo.com', 'user@localhost']), sorted(to))