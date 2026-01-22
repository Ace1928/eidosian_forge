import sys
from typing import List, Optional, Union
from twisted.conch.ssh.transport import SSHCiphers, SSHClientTransport
from twisted.python import usage
def opt_user_authentications(self, uas):
    """Choose how to authenticate to the remote server"""
    if isinstance(uas, str):
        uas = uas.encode('utf-8')
    self['user-authentications'] = uas.split(b',')