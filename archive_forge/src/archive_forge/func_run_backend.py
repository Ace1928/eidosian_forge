import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
def run_backend(self):
    """Call out to git http-backend."""
    rest = self.path
    i = rest.rfind('?')
    if i >= 0:
        rest, query = (rest[:i], rest[i + 1:])
    else:
        query = ''
    env = copy.deepcopy(os.environ)
    env['SERVER_SOFTWARE'] = self.version_string()
    env['SERVER_NAME'] = self.server.server_name
    env['GATEWAY_INTERFACE'] = 'CGI/1.1'
    env['SERVER_PROTOCOL'] = self.protocol_version
    env['SERVER_PORT'] = str(self.server.server_port)
    env['GIT_PROJECT_ROOT'] = self.server.root_path
    env['GIT_HTTP_EXPORT_ALL'] = '1'
    env['REQUEST_METHOD'] = self.command
    uqrest = unquote(rest)
    env['PATH_INFO'] = uqrest
    env['SCRIPT_NAME'] = '/'
    if query:
        env['QUERY_STRING'] = query
    host = self.address_string()
    if host != self.client_address[0]:
        env['REMOTE_HOST'] = host
    env['REMOTE_ADDR'] = self.client_address[0]
    authorization = self.headers.get('authorization')
    if authorization:
        authorization = authorization.split()
        if len(authorization) == 2:
            import base64
            import binascii
            env['AUTH_TYPE'] = authorization[0]
            if authorization[0].lower() == 'basic':
                try:
                    authorization = base64.decodestring(authorization[1])
                except binascii.Error:
                    pass
                else:
                    authorization = authorization.split(':')
                    if len(authorization) == 2:
                        env['REMOTE_USER'] = authorization[0]
    content_type = self.headers.get('content-type')
    if content_type:
        env['CONTENT_TYPE'] = content_type
    length = self.headers.get('content-length')
    if length:
        env['CONTENT_LENGTH'] = length
    referer = self.headers.get('referer')
    if referer:
        env['HTTP_REFERER'] = referer
    accept = []
    for line in self.headers.getallmatchingheaders('accept'):
        if line[:1] in '\t\n\r ':
            accept.append(line.strip())
        else:
            accept = accept + line[7:].split(',')
    env['HTTP_ACCEPT'] = ','.join(accept)
    ua = self.headers.get('user-agent')
    if ua:
        env['HTTP_USER_AGENT'] = ua
    co = self.headers.get('cookie')
    if co:
        env['HTTP_COOKIE'] = co
    for k in ('QUERY_STRING', 'REMOTE_HOST', 'CONTENT_LENGTH', 'HTTP_USER_AGENT', 'HTTP_COOKIE', 'HTTP_REFERER'):
        env.setdefault(k, '')
    self.wfile.write(b'HTTP/1.1 200 Script output follows\r\n')
    self.wfile.write(('Server: %s\r\n' % self.server.server_name).encode('ascii'))
    self.wfile.write(('Date: %s\r\n' % self.date_time_string()).encode('ascii'))
    decoded_query = query.replace('+', ' ')
    try:
        nbytes = int(length)
    except (TypeError, ValueError):
        nbytes = -1
    if self.command.lower() == 'post':
        if nbytes > 0:
            data = self.rfile.read(nbytes)
        elif self.headers.get('transfer-encoding') == 'chunked':
            chunks = []
            while True:
                line = self.rfile.readline()
                length = int(line.rstrip(), 16)
                chunk = self.rfile.read(length + 2)
                chunks.append(chunk[:-2])
                if length == 0:
                    break
            data = b''.join(chunks)
            env['CONTENT_LENGTH'] = str(len(data))
        else:
            raise AssertionError
    else:
        data = None
        env['CONTENT_LENGTH'] = '0'
    while select.select([self.rfile._sock], [], [], 0)[0]:
        if not self.rfile._sock.recv(1):
            break
    args = ['http-backend']
    if '=' not in decoded_query:
        args.append(decoded_query)
    stdout = run_git_or_fail(args, input=data, env=env, stderr=subprocess.PIPE)
    self.wfile.write(stdout)