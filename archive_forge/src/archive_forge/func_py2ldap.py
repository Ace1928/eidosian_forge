import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def py2ldap(val):
    """Type convert a Python value to a type accepted by LDAP (unicode).

    The LDAP API only accepts strings for values therefore convert
    the value's type to a unicode string. A subsequent type conversion
    will encode the unicode as UTF-8 as required by the python-ldap API,
    but for now we just want a string representation of the value.

    :param val: The value to convert to a LDAP string representation
    :returns: unicode string representation of value.
    """
    if isinstance(val, bool):
        return u'TRUE' if val else u'FALSE'
    else:
        return str(val)