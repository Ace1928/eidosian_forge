import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_multiple_destinations(self):
    to_addresses = ['to1@to.com', 'to2@to.com', 'to3@to.com']
    msg = EmailMessage('from@from.com', to_addresses, 'subject')
    self.assertContainsRe(msg.as_string(), 'To: ' + ', '.join(to_addresses))