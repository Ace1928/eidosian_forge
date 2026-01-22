import sys
from typing import List, Optional, Union
from twisted.conch.ssh.transport import SSHCiphers, SSHClientTransport
from twisted.python import usage
def opt_macs(self, macs):
    """Specify MAC algorithms"""
    if isinstance(macs, str):
        macs = macs.encode('utf-8')
    macs = macs.split(b',')
    for mac in macs:
        if mac not in SSHCiphers.macMap:
            sys.exit("Unknown mac type '%r'" % mac)
    self['macs'] = macs