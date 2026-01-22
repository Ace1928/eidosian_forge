import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
def get_allowed_auths(self, username):
    return 'password,publickey'