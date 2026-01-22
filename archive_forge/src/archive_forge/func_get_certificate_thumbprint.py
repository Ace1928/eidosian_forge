import base64
import collections.abc
import contextlib
import grp
import hashlib
import itertools
import os
import pwd
import uuid
from cryptography import x509
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import urllib
from keystone.common import password_hashing
import keystone.conf
from keystone import exception
from keystone.i18n import _
def get_certificate_thumbprint(cert_pem):
    """Get certificate thumbprint from the PEM certificate content.

    :param str cert_pem: the PEM certificate content
    :rtype: certificate thumbprint
    """
    thumb_sha256 = hashlib.sha256(cert_pem.encode('ascii')).digest()
    thumbprint = base64.urlsafe_b64encode(thumb_sha256).decode('ascii')
    return thumbprint