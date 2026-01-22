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
def result3(self, msgid=ldap.RES_ANY, all=1, timeout=None, resp_ctrl_classes=None):
    ldap_result = self.conn.result3(msgid, all, timeout, resp_ctrl_classes)
    LOG.debug('LDAP result3: msgid=%s all=%s timeout=%s resp_ctrl_classes=%s ldap_result=%s', msgid, all, timeout, resp_ctrl_classes, ldap_result)
    rtype, rdata, rmsgid, serverctrls = ldap_result
    py_result = convert_ldap_result(rdata)
    return py_result