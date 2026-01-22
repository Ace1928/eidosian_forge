import hmac
import sys
from binascii import Error as DecodeError, a2b_base64, b2a_base64
from contextlib import closing
from hashlib import sha1
from zope.interface import implementer
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.conch.ssh.keys import BadKeyError, FingerprintFormats, Key
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString
from twisted.python.randbytes import secureRandom
from twisted.python.util import FancyEqMixin
def gotHasKey(result):
    if result:
        if not self.hasHostKey(ip, key):
            ui.warn("Warning: Permanently added the %s host key for IP address '%s' to the list of known hosts." % (key.type(), nativeString(ip)))
            self.addHostKey(ip, key)
            self.save()
        return result
    else:

        def promptResponse(response):
            if response:
                self.addHostKey(hostname, key)
                self.addHostKey(ip, key)
                self.save()
                return response
            else:
                raise UserRejectedKey()
        keytype = key.type()
        if keytype == 'EC':
            keytype = 'ECDSA'
        prompt = "The authenticity of host '%s (%s)' can't be established.\n%s key fingerprint is SHA256:%s.\nAre you sure you want to continue connecting (yes/no)? " % (nativeString(hostname), nativeString(ip), keytype, key.fingerprint(format=FingerprintFormats.SHA256_BASE64))
        proceed = ui.prompt(prompt.encode(sys.getdefaultencoding()))
        return proceed.addCallback(promptResponse)