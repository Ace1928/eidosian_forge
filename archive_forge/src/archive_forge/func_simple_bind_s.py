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
def simple_bind_s(self, who='', cred='', serverctrls=None, clientctrls=None):
    LOG.debug('LDAP bind: who=%s', who)
    return self.conn.simple_bind_s(who, cred, serverctrls=serverctrls, clientctrls=clientctrls)