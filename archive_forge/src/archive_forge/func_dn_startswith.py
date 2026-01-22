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
def dn_startswith(descendant_dn, dn):
    """Return True if and only if the descendant_dn is under the dn.

    :param descendant_dn: Either a string DN or a DN parsed by ldap.dn.str2dn.
    :param dn: Either a string DN or a DN parsed by ldap.dn.str2dn.

    """
    if not isinstance(descendant_dn, list):
        descendant_dn = ldap.dn.str2dn(descendant_dn)
    if not isinstance(dn, list):
        dn = ldap.dn.str2dn(dn)
    if len(descendant_dn) <= len(dn):
        return False
    return is_dn_equal(descendant_dn[-len(dn):], dn)