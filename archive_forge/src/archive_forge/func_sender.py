from __future__ import print_function
import email.utils
import errno
import hashlib
import mimetypes
import os
import re
import base64
import binascii
import math
from hashlib import md5
import boto.utils
from boto.compat import BytesIO, six, urllib, encodebytes
from boto.exception import BotoClientError
from boto.exception import StorageDataError
from boto.exception import PleaseRetryException
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.provider import Provider
from boto.s3.keyfile import KeyFile
from boto.s3.user import User
from boto import UserAgent
import boto.utils
from boto.utils import compute_md5, compute_hash
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import print_to_fd
def sender(http_conn, method, path, data, headers):
    if spos is not None and spos != fp.tell():
        fp.seek(spos)
    elif spos is None and self.read_from_stream:
        raise provider.storage_data_error('Cannot retry failed request. fp does not support seeking.')
    skips = {}
    if boto.utils.find_matching_headers('host', headers):
        skips['skip_host'] = 1
    if boto.utils.find_matching_headers('accept-encoding', headers):
        skips['skip_accept_encoding'] = 1
    http_conn.putrequest(method, path, **skips)
    for key in headers:
        http_conn.putheader(key, headers[key])
    http_conn.endheaders()
    save_debug = self.bucket.connection.debug
    self.bucket.connection.debug = 0
    if getattr(http_conn, 'debuglevel', 0) < 4:
        http_conn.set_debuglevel(0)
    data_len = 0
    if cb:
        if size:
            cb_size = size
        elif self.size:
            cb_size = self.size
        else:
            cb_size = 0
        if chunked_transfer and cb_size == 0:
            cb_count = 1024 * 1024 / self.BufferSize
        elif num_cb > 1:
            cb_count = int(math.ceil(cb_size / self.BufferSize / (num_cb - 1.0)))
        elif num_cb < 0:
            cb_count = -1
        else:
            cb_count = 0
        i = 0
        cb(data_len, cb_size)
    bytes_togo = size
    if bytes_togo and bytes_togo < self.BufferSize:
        chunk = fp.read(bytes_togo)
    else:
        chunk = fp.read(self.BufferSize)
    if not isinstance(chunk, bytes):
        chunk = chunk.encode('utf-8')
    if spos is None:
        self.read_from_stream = True
    while chunk:
        chunk_len = len(chunk)
        data_len += chunk_len
        if chunked_transfer:
            chunk_len_bytes = ('%x' % chunk_len).encode('utf-8')
            http_conn.send(chunk_len_bytes + b';\r\n')
            http_conn.send(chunk)
            http_conn.send(b'\r\n')
        else:
            http_conn.send(chunk)
        for alg in digesters:
            digesters[alg].update(chunk)
        if bytes_togo:
            bytes_togo -= chunk_len
            if bytes_togo <= 0:
                break
        if cb:
            i += 1
            if i == cb_count or cb_count == -1:
                cb(data_len, cb_size)
                i = 0
        if bytes_togo and bytes_togo < self.BufferSize:
            chunk = fp.read(bytes_togo)
        else:
            chunk = fp.read(self.BufferSize)
        if not isinstance(chunk, bytes):
            chunk = chunk.encode('utf-8')
    self.size = data_len
    for alg in digesters:
        self.local_hashes[alg] = digesters[alg].digest()
    if chunked_transfer:
        http_conn.send(b'0\r\n')
        http_conn.send(b'\r\n')
    if cb and (cb_count <= 1 or i > 0) and (data_len > 0):
        cb(data_len, cb_size)
    http_conn.set_debuglevel(save_debug)
    self.bucket.connection.debug = save_debug
    response = http_conn.getresponse()
    body = response.read()
    if not self.should_retry(response, chunked_transfer):
        raise provider.storage_response_error(response.status, response.reason, body)
    return response