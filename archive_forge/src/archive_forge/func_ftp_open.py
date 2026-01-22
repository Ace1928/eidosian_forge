import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def ftp_open(self, req):
    import ftplib
    import mimetypes
    host = req.host
    if not host:
        raise URLError('ftp error: no host given')
    host, port = _splitport(host)
    if port is None:
        port = ftplib.FTP_PORT
    else:
        port = int(port)
    user, host = _splituser(host)
    if user:
        user, passwd = _splitpasswd(user)
    else:
        passwd = None
    host = unquote(host)
    user = user or ''
    passwd = passwd or ''
    try:
        host = socket.gethostbyname(host)
    except OSError as msg:
        raise URLError(msg)
    path, attrs = _splitattr(req.selector)
    dirs = path.split('/')
    dirs = list(map(unquote, dirs))
    dirs, file = (dirs[:-1], dirs[-1])
    if dirs and (not dirs[0]):
        dirs = dirs[1:]
    try:
        fw = self.connect_ftp(user, passwd, host, port, dirs, req.timeout)
        type = file and 'I' or 'D'
        for attr in attrs:
            attr, value = _splitvalue(attr)
            if attr.lower() == 'type' and value in ('a', 'A', 'i', 'I', 'd', 'D'):
                type = value.upper()
        fp, retrlen = fw.retrfile(file, type)
        headers = ''
        mtype = mimetypes.guess_type(req.full_url)[0]
        if mtype:
            headers += 'Content-type: %s\n' % mtype
        if retrlen is not None and retrlen >= 0:
            headers += 'Content-length: %d\n' % retrlen
        headers = email.message_from_string(headers)
        return addinfourl(fp, headers, req.full_url)
    except ftplib.all_errors as exp:
        raise URLError(exp) from exp