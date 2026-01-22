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
def docmd(self, cmd, args=''):
    """Send a command, and return its response code."""
    self.putcmd(cmd, args)
    return self.getreply()