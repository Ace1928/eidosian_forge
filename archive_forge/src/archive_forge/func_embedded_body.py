import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def embedded_body(lines):
    return linesep.join(lines) + linesep