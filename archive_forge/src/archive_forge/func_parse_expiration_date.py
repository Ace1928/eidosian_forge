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
def parse_expiration_date(expiration_date):
    if not expiration_date.endswith('Z'):
        expiration_date += 'Z'
    try:
        expiration_time = timeutils.parse_isotime(expiration_date)
    except ValueError:
        raise exception.ValidationTimeStampError()
    if timeutils.is_older_than(expiration_time, 0):
        raise exception.ValidationExpirationError()
    return expiration_time