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
def is_dn_equal(dn1, dn2):
    """Return True if and only if the DNs are equal.

    Two DNs are equal if they've got the same number of RDNs and if the RDNs
    are the same at each position. See RFC4517.

    Note that this function uses is_rdn_equal to compare RDNs so the
    limitations of that function apply here.

    :param dn1: Either a string DN or a DN parsed by ldap.dn.str2dn.
    :param dn2: Either a string DN or a DN parsed by ldap.dn.str2dn.

    """
    if not isinstance(dn1, list):
        dn1 = ldap.dn.str2dn(dn1)
    if not isinstance(dn2, list):
        dn2 = ldap.dn.str2dn(dn2)
    if len(dn1) != len(dn2):
        return False
    for rdn1, rdn2 in zip(dn1, dn2):
        if not is_rdn_equal(rdn1, rdn2):
            return False
    return True