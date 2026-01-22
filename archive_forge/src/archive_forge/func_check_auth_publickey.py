import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
def check_auth_publickey(self, username, key):
    pubkey = paramiko.RSAKey.from_private_key(StringIO(CLIENT_KEY))
    if username == USER and key == pubkey:
        return paramiko.AUTH_SUCCESSFUL
    return paramiko.AUTH_FAILED