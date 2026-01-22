import copy
import datetime
import email.utils
import html
import http.client
import io
import itertools
import mimetypes
import os
import posixpath
import select
import shutil
import socket # For gethostbyaddr()
import socketserver
import sys
import time
import urllib.parse
from http import HTTPStatus
def run_cgi(self):
    """Execute a CGI script."""
    dir, rest = self.cgi_info
    path = dir + '/' + rest
    i = path.find('/', len(dir) + 1)
    while i >= 0:
        nextdir = path[:i]
        nextrest = path[i + 1:]
        scriptdir = self.translate_path(nextdir)
        if os.path.isdir(scriptdir):
            dir, rest = (nextdir, nextrest)
            i = path.find('/', len(dir) + 1)
        else:
            break
    rest, _, query = rest.partition('?')
    i = rest.find('/')
    if i >= 0:
        script, rest = (rest[:i], rest[i:])
    else:
        script, rest = (rest, '')
    scriptname = dir + '/' + script
    scriptfile = self.translate_path(scriptname)
    if not os.path.exists(scriptfile):
        self.send_error(HTTPStatus.NOT_FOUND, 'No such CGI script (%r)' % scriptname)
        return
    if not os.path.isfile(scriptfile):
        self.send_error(HTTPStatus.FORBIDDEN, 'CGI script is not a plain file (%r)' % scriptname)
        return
    ispy = self.is_python(scriptname)
    if self.have_fork or not ispy:
        if not self.is_executable(scriptfile):
            self.send_error(HTTPStatus.FORBIDDEN, 'CGI script is not executable (%r)' % scriptname)
            return
    env = copy.deepcopy(os.environ)
    env['SERVER_SOFTWARE'] = self.version_string()
    env['SERVER_NAME'] = self.server.server_name
    env['GATEWAY_INTERFACE'] = 'CGI/1.1'
    env['SERVER_PROTOCOL'] = self.protocol_version
    env['SERVER_PORT'] = str(self.server.server_port)
    env['REQUEST_METHOD'] = self.command
    uqrest = urllib.parse.unquote(rest)
    env['PATH_INFO'] = uqrest
    env['PATH_TRANSLATED'] = self.translate_path(uqrest)
    env['SCRIPT_NAME'] = scriptname
    env['QUERY_STRING'] = query
    env['REMOTE_ADDR'] = self.client_address[0]
    authorization = self.headers.get('authorization')
    if authorization:
        authorization = authorization.split()
        if len(authorization) == 2:
            import base64, binascii
            env['AUTH_TYPE'] = authorization[0]
            if authorization[0].lower() == 'basic':
                try:
                    authorization = authorization[1].encode('ascii')
                    authorization = base64.decodebytes(authorization).decode('ascii')
                except (binascii.Error, UnicodeError):
                    pass
                else:
                    authorization = authorization.split(':')
                    if len(authorization) == 2:
                        env['REMOTE_USER'] = authorization[0]
    if self.headers.get('content-type') is None:
        env['CONTENT_TYPE'] = self.headers.get_content_type()
    else:
        env['CONTENT_TYPE'] = self.headers['content-type']
    length = self.headers.get('content-length')
    if length:
        env['CONTENT_LENGTH'] = length
    referer = self.headers.get('referer')
    if referer:
        env['HTTP_REFERER'] = referer
    accept = self.headers.get_all('accept', ())
    env['HTTP_ACCEPT'] = ','.join(accept)
    ua = self.headers.get('user-agent')
    if ua:
        env['HTTP_USER_AGENT'] = ua
    co = filter(None, self.headers.get_all('cookie', []))
    cookie_str = ', '.join(co)
    if cookie_str:
        env['HTTP_COOKIE'] = cookie_str
    for k in ('QUERY_STRING', 'REMOTE_HOST', 'CONTENT_LENGTH', 'HTTP_USER_AGENT', 'HTTP_COOKIE', 'HTTP_REFERER'):
        env.setdefault(k, '')
    self.send_response(HTTPStatus.OK, 'Script output follows')
    self.flush_headers()
    decoded_query = query.replace('+', ' ')
    if self.have_fork:
        args = [script]
        if '=' not in decoded_query:
            args.append(decoded_query)
        nobody = nobody_uid()
        self.wfile.flush()
        pid = os.fork()
        if pid != 0:
            pid, sts = os.waitpid(pid, 0)
            while select.select([self.rfile], [], [], 0)[0]:
                if not self.rfile.read(1):
                    break
            exitcode = os.waitstatus_to_exitcode(sts)
            if exitcode:
                self.log_error(f'CGI script exit code {exitcode}')
            return
        try:
            try:
                os.setuid(nobody)
            except OSError:
                pass
            os.dup2(self.rfile.fileno(), 0)
            os.dup2(self.wfile.fileno(), 1)
            os.execve(scriptfile, args, env)
        except:
            self.server.handle_error(self.request, self.client_address)
            os._exit(127)
    else:
        import subprocess
        cmdline = [scriptfile]
        if self.is_python(scriptfile):
            interp = sys.executable
            if interp.lower().endswith('w.exe'):
                interp = interp[:-5] + interp[-4:]
            cmdline = [interp, '-u'] + cmdline
        if '=' not in query:
            cmdline.append(query)
        self.log_message('command: %s', subprocess.list2cmdline(cmdline))
        try:
            nbytes = int(length)
        except (TypeError, ValueError):
            nbytes = 0
        p = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if self.command.lower() == 'post' and nbytes > 0:
            data = self.rfile.read(nbytes)
        else:
            data = None
        while select.select([self.rfile._sock], [], [], 0)[0]:
            if not self.rfile._sock.recv(1):
                break
        stdout, stderr = p.communicate(data)
        self.wfile.write(stdout)
        if stderr:
            self.log_error('%s', stderr)
        p.stderr.close()
        p.stdout.close()
        status = p.returncode
        if status:
            self.log_error('CGI script exit status %#x', status)
        else:
            self.log_message('CGI script exited OK')