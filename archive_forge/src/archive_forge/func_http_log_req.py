import argparse
import functools
import hashlib
import logging
import os
from oslo_utils import encodeutils
from oslo_utils import importutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
def http_log_req(_logger, args, kwargs):
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    string_parts = ['curl -i']
    for element in args:
        if element in ('GET', 'POST', 'DELETE', 'PUT'):
            string_parts.append(' -X %s' % element)
        else:
            string_parts.append(' %s' % element)
    for key, value in kwargs['headers'].items():
        if key in SENSITIVE_HEADERS:
            v = value.encode('utf-8')
            h = hashlib.sha256(v)
            d = h.hexdigest()
            value = '{SHA256}%s' % d
        header = ' -H "%s: %s"' % (key, value)
        string_parts.append(header)
    if 'body' in kwargs and kwargs['body']:
        string_parts.append(" -d '%s'" % kwargs['body'])
    req = encodeutils.safe_encode(''.join(string_parts))
    _logger.debug('REQ: %s', req)