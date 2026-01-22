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
def parse_deref(opt):
    try:
        return LDAP_DEREF[opt]
    except KeyError:
        raise ValueError(_('Invalid LDAP deref option: %(option)s. Choose one of: %(options)s') % {'option': opt, 'options': ', '.join(LDAP_DEREF.keys())})