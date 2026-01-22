import errno
import os
import re
import socket
import sys
from datetime import datetime
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.servers.basehttp import WSGIServer, get_internal_wsgi_application, run
from django.utils import autoreload
from django.utils.regex_helper import _lazy_re_compile
def on_bind(self, server_port):
    quit_command = 'CTRL-BREAK' if sys.platform == 'win32' else 'CONTROL-C'
    if self._raw_ipv6:
        addr = f'[{self.addr}]'
    elif self.addr == '0':
        addr = '0.0.0.0'
    else:
        addr = self.addr
    now = datetime.now().strftime('%B %d, %Y - %X')
    version = self.get_version()
    print(f'{now}\nDjango version {version}, using settings {settings.SETTINGS_MODULE!r}\nStarting development server at {self.protocol}://{addr}:{server_port}/\nQuit the server with {quit_command}.', file=self.stdout)