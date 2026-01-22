from zope.interface import implementer
from twisted.application import service, strports
from twisted.conch import manhole, manhole_ssh, telnet
from twisted.conch.insults import insults
from twisted.conch.ssh import keys
from twisted.cred import checkers, portal
from twisted.internet import protocol
from twisted.python import filepath, usage

    Create a manhole server service.

    @type options: L{dict}
    @param options: A mapping describing the configuration of
    the desired service.  Recognized key/value pairs are::

        "telnetPort": strports description of the address on which
                      to listen for telnet connections.  If None,
                      no telnet service will be started.

        "sshPort": strports description of the address on which to
                   listen for ssh connections.  If None, no ssh
                   service will be started.

        "namespace": dictionary containing desired initial locals
                     for manhole connections.  If None, an empty
                     dictionary will be used.

        "passwd": Name of a passwd(5)-format username/password file.

        "sshKeyDir": The folder that the SSH server key will be kept in.

        "sshKeyName": The filename of the key.

        "sshKeySize": The size of the key, in bits. Default is 4096.

    @rtype: L{twisted.application.service.IService}
    @return: A manhole service.
    