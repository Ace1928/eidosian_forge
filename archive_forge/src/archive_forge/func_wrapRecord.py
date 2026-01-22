import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def wrapRecord(self, type):

    def wrapRecordFunc(name, *arg, **kw):
        return (dns.domainString(name), type(*arg, **kw))
    return wrapRecordFunc