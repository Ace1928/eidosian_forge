import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_multipart_message_simple(self):
    msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
    msg.add_inline_attachment('body')
    self.assertEqualDiff(simple_multipart_message(), msg.as_string(BOUNDARY))