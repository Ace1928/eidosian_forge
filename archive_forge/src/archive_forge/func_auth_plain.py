import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
def auth_plain(self, challenge=None):
    """ Authobject to use with PLAIN authentication. Requires self.user and
        self.password to be set."""
    return '\x00%s\x00%s' % (self.user, self.password)