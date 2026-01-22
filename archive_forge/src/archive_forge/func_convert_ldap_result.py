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
def convert_ldap_result(ldap_result):
    """Convert LDAP search result to Python types used by OpenStack.

    Each result tuple is of the form (dn, attrs), where dn is a string
    containing the DN (distinguished name) of the entry, and attrs is
    a dictionary containing the attributes associated with the
    entry. The keys of attrs are strings, and the associated values
    are lists of strings.

    OpenStack wants to use Python types of its choosing. Strings will
    be unicode, truth values boolean, whole numbers int's, etc. DN's are
    represented as text in python-ldap by default for Python 3 and when
    bytes_mode=False for Python 2, and therefore do not require decoding.

    :param ldap_result: LDAP search result
    :returns: list of 2-tuples containing (dn, attrs) where dn is unicode
              and attrs is a dict whose values are type converted to
              OpenStack preferred types.
    """
    py_result = []
    at_least_one_referral = False
    for dn, attrs in ldap_result:
        ldap_attrs = {}
        if dn is None:
            at_least_one_referral = True
            continue
        for kind, values in attrs.items():
            try:
                val2py = enabled2py if kind == 'enabled' else ldap2py
                ldap_attrs[kind] = [val2py(x) for x in values]
            except UnicodeDecodeError:
                LOG.debug('Unable to decode value for attribute %s', kind)
        py_result.append((dn, ldap_attrs))
    if at_least_one_referral:
        LOG.debug('Referrals were returned and ignored. Enable referral chasing in keystone.conf via [ldap] chase_referrals')
    return py_result