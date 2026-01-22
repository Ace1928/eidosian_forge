import subprocess
import time
import logging.handlers
import boto
import boto.provider
import collections
import tempfile
import random
import smtplib
import datetime
import re
import io
import email.mime.multipart
import email.mime.base
import email.mime.text
import email.utils
import email.encoders
import gzip
import threading
import locale
import sys
from boto.compat import six, StringIO, urllib, encodebytes
from contextlib import contextmanager
from hashlib import md5, sha512
from boto.compat import json
def write_mime_multipart(content, compress=False, deftype='text/plain', delimiter=':'):
    """Description:
    :param content: A list of tuples of name-content pairs. This is used
    instead of a dict to ensure that scripts run in order
    :type list of tuples:

    :param compress: Use gzip to compress the scripts, defaults to no compression
    :type bool:

    :param deftype: The type that should be assumed if nothing else can be figured out
    :type str:

    :param delimiter: mime delimiter
    :type str:

    :return: Final mime multipart
    :rtype: str:
    """
    wrapper = email.mime.multipart.MIMEMultipart()
    for name, con in content:
        definite_type = guess_mime_type(con, deftype)
        maintype, subtype = definite_type.split('/', 1)
        if maintype == 'text':
            mime_con = email.mime.text.MIMEText(con, _subtype=subtype)
        else:
            mime_con = email.mime.base.MIMEBase(maintype, subtype)
            mime_con.set_payload(con)
            email.encoders.encode_base64(mime_con)
        mime_con.add_header('Content-Disposition', 'attachment', filename=name)
        wrapper.attach(mime_con)
    rcontent = wrapper.as_string()
    if compress:
        buf = StringIO()
        gz = gzip.GzipFile(mode='wb', fileobj=buf)
        try:
            gz.write(rcontent)
        finally:
            gz.close()
        rcontent = buf.getvalue()
    return rcontent