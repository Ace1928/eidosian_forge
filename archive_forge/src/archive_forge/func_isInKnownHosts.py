import contextlib
import getpass
import io
import os
import sys
from base64 import decodebytes
from twisted.conch.client import agent
from twisted.conch.client.knownhosts import ConsoleUI, KnownHostsFile
from twisted.conch.error import ConchError
from twisted.conch.ssh import common, keys, userauth
from twisted.internet import defer, protocol, reactor
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
def isInKnownHosts(host, pubKey, options):
    """
    Checks to see if host is in the known_hosts file for the user.

    @return: 0 if it isn't, 1 if it is and is the same, 2 if it's changed.
    @rtype: L{int}
    """
    keyType = common.getNS(pubKey)[0]
    retVal = 0
    if not options['known-hosts'] and (not os.path.exists(os.path.expanduser('~/.ssh/'))):
        print('Creating ~/.ssh directory...')
        os.mkdir(os.path.expanduser('~/.ssh'))
    kh_file = options['known-hosts'] or _KNOWN_HOSTS
    try:
        known_hosts = open(os.path.expanduser(kh_file), 'rb')
    except OSError:
        return 0
    with known_hosts:
        for line in known_hosts.readlines():
            split = line.split()
            if len(split) < 3:
                continue
            hosts, hostKeyType, encodedKey = split[:3]
            if host not in hosts.split(b','):
                continue
            if hostKeyType != keyType:
                continue
            try:
                decodedKey = decodebytes(encodedKey)
            except BaseException:
                continue
            if decodedKey == pubKey:
                return 1
            else:
                retVal = 2
    return retVal