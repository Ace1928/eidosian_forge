from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def get_cert_fingerprint(self, fqdn, port, proxy_host=None, proxy_port=None):
    if proxy_host:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect((proxy_host, proxy_port))
        command = 'CONNECT %s:%d HTTP/1.0\r\n\r\n' % (fqdn, port)
        sock.send(command.encode())
        buf = sock.recv(8192).decode()
        if buf.split()[1] != '200':
            self.module.fail_json(msg='Failed to connect to the proxy')
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        der_cert_bin = ctx.wrap_socket(sock, server_hostname=fqdn).getpeercert(True)
        sock.close()
    else:
        try:
            pem = ssl.get_server_certificate((fqdn, port))
        except Exception:
            self.module.fail_json(msg=f'Cannot connect to host: {fqdn}')
        der_cert_bin = ssl.PEM_cert_to_DER_cert(pem)
    if der_cert_bin:
        string = str(hashlib.sha1(der_cert_bin).hexdigest())
        return ':'.join((a + b for a, b in zip(string[::2], string[1::2])))
    else:
        self.module.fail_json(msg=f'Unable to obtain certificate fingerprint for host: {fqdn}')