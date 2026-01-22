from __future__ import annotations
import os
import socket
import string
from queue import Empty
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
@property
def slmq(self):
    if self._slmq is None:
        conninfo = self.conninfo
        account = os.environ.get('SLMQ_ACCOUNT', conninfo.virtual_host)
        user = os.environ.get('SL_USERNAME', conninfo.userid)
        api_key = os.environ.get('SL_API_KEY', conninfo.password)
        host = os.environ.get('SLMQ_HOST', conninfo.hostname)
        port = os.environ.get('SLMQ_PORT', conninfo.port)
        secure = bool(os.environ.get('SLMQ_SECURE', self.transport_options.get('secure')) or True)
        endpoint = '{}://{}{}'.format('https' if secure else 'http', host, f':{port}' if port else '')
        self._slmq = get_client(account, endpoint=endpoint)
        self._slmq.authenticate(user, api_key)
    return self._slmq