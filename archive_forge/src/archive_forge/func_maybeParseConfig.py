import errno
import os
import warnings
from zope.interface import moduleProvides
from twisted.internet import defer, error, interfaces, protocol
from twisted.internet.abstract import isIPv6Address
from twisted.names import cache, common, dns, hosts as hostsModule, resolve, root
from twisted.python import failure, log
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.internet.base import ThreadedResolver as _ThreadedResolverImpl
def maybeParseConfig(self):
    if self.resolv is None:
        return
    try:
        resolvConf = self._openFile(self.resolv)
    except OSError as e:
        if e.errno == errno.ENOENT:
            self.parseConfig(())
        else:
            raise
    else:
        with resolvConf:
            mtime = os.fstat(resolvConf.fileno()).st_mtime
            if mtime != self._lastResolvTime:
                log.msg(f'{self.resolv} changed, reparsing')
                self._lastResolvTime = mtime
                self.parseConfig(resolvConf)
    self._parseCall = self._reactor.callLater(self._resolvReadInterval, self.maybeParseConfig)