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
def search_ext(self, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0, serverctrls=None, clientctrls=None, timeout=-1, sizelimit=0):
    if attrlist is not None:
        attrlist = [attr for attr in attrlist if attr is not None]
    LOG.debug('LDAP search_ext: base=%s scope=%s filterstr=%s attrs=%s attrsonly=%s serverctrls=%s clientctrls=%s timeout=%s sizelimit=%s', base, scope, filterstr, attrlist, attrsonly, serverctrls, clientctrls, timeout, sizelimit)
    return self.conn.search_ext(base, scope, filterstr, attrlist, attrsonly, serverctrls, clientctrls, timeout, sizelimit)