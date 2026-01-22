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
def get_certificate_subject_dn(cert_pem):
    """Get subject DN from the PEM certificate content.

    :param str cert_pem: the PEM certificate content
    :rtype: JSON data for subject DN
    :raises keystone.exception.ValidationError: if the PEM certificate content
        is invalid
    """
    dn_dict = {}
    try:
        cert = x509.load_pem_x509_certificate(cert_pem.encode('utf-8'))
        for item in cert.subject:
            name, value = item.rfc4514_string().split('=')
            if item.oid in ATTR_NAME_OVERRIDES:
                name = ATTR_NAME_OVERRIDES[item.oid]
            dn_dict[name] = value
    except Exception as error:
        LOG.exception(error)
        message = _('The certificate content is not PEM format.')
        raise exception.ValidationError(message=message)
    return dn_dict