import logging
import socket
from subprocess import PIPE
from subprocess import Popen
import sys
import time
import traceback
import requests
from saml2test.check import CRITICAL
def ip_addresses():
    return [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith('127.')]