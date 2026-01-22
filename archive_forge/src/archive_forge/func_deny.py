import email.utils
import getpass
import os
import sys
from configparser import ConfigParser
from io import StringIO
from twisted.copyright import version
from twisted.internet import reactor
from twisted.logger import Logger, textFileLogObserver
from twisted.mail import smtp
def deny(conf):
    uid = os.getuid()
    gid = os.getgid()
    if conf.useraccess == 'deny':
        if uid in conf.denyUIDs:
            return True
        if uid in conf.allowUIDs:
            return False
    else:
        if uid in conf.allowUIDs:
            return False
        if uid in conf.denyUIDs:
            return True
    if conf.groupaccess == 'deny':
        if gid in conf.denyGIDs:
            return True
        if gid in conf.allowGIDs:
            return False
    else:
        if gid in conf.allowGIDs:
            return False
        if gid in conf.denyGIDs:
            return True
    return not conf.defaultAccess