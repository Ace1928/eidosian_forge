import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def get_text_content(msg, errors='replace'):
    content = msg.get_payload(decode=True)
    charset = msg.get_param('charset', 'ASCII')
    return content.decode(charset, errors=errors)