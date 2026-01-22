import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
def check_auth_password(self, username, password):
    if username == USER and password == PASSWORD:
        return paramiko.AUTH_SUCCESSFUL
    return paramiko.AUTH_FAILED