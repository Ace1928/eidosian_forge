import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_setting_headers(self):
    msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
    msg['To'] = 'to2@to.com'
    msg['Cc'] = 'cc@cc.com'
    self.assertEqual('to2@to.com', msg['To'])
    self.assertEqual('cc@cc.com', msg['Cc'])