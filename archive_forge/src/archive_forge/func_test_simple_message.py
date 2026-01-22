import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_simple_message(self):
    pairs = {b'body': SIMPLE_MESSAGE_ASCII, 'bódy': SIMPLE_MESSAGE_UTF8, b'b\xc3\xb3dy': SIMPLE_MESSAGE_UTF8, b'b\xf4dy': SIMPLE_MESSAGE_8BIT}
    for body, expected in pairs.items():
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject', body)
        self.assertEqualDiff(expected, msg.as_string())