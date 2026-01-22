import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def monitorFile(self, name, callback, interval=10):
    """
        Start monitoring a file for changes.

        @type name: L{bytes}
        @param name: The name of a file to monitor.

        @type callback: callable which takes a L{bytes} argument
        @param callback: The function to call when the file has changed.

        @type interval: L{float}
        @param interval: The interval in seconds between checks.
        """
    try:
        mtime = os.path.getmtime(name)
    except BaseException:
        mtime = 0
    self.files.append([interval, name, callback, mtime])
    self.intervals.addInterval(interval)