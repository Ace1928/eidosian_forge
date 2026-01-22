import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_string_with_encoding(self):
    pairs = {'Pepe': (b'Pepe', 'ascii'), 'Pérez': (b'P\xc3\xa9rez', 'utf-8'), b'P\xc3\xa9rez': (b'P\xc3\xa9rez', 'utf-8'), b'P\xe8rez': (b'P\xe8rez', '8-bit')}
    for string_, pair in pairs.items():
        self.assertEqual(pair, EmailMessage.string_with_encoding(string_))