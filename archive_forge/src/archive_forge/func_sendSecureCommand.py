from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def sendSecureCommand(self, server, keyfile, certfile, url, response, follow_redirects=1):
    """posts the protocol buffer via https to the desired url on the server,
    using the specified key and certificate files, and puts the return
    data int othe protocol buffer 'response'.

    See caveats in sendCommand.

    You need an SSL-aware build of the Python2 interpreter to use this command.
    (Python1 is not supported).  An SSL build of python2.2 is in
    /home/build/buildtools/python-ssl-2.2 . An SSL build of python is
    standard on all prod machines.

    keyfile: Contains our private RSA key
    certfile: Contains SSL certificate for remote host
    Specify None for keyfile/certfile if you don't want to do client auth.
    """
    return self.sendCommand(server, url, response, follow_redirects=follow_redirects, secure=1, keyfile=keyfile, certfile=certfile)