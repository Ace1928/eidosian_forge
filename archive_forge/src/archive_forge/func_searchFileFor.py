from twisted.internet import defer
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.names import common, dns
from twisted.python import failure
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
def searchFileFor(file, name):
    """
    Grep given file, which is in hosts(5) standard format, for an address
    entry with a given name.

    @param file: The name of the hosts(5)-format file to search.
    @type file: C{str} or C{bytes}

    @param name: The name to search for.
    @type name: C{bytes}

    @return: L{None} if the name is not found in the file, otherwise a
        C{str} giving the first address in the file associated with
        the name.
    """
    addresses = searchFileForAll(FilePath(file), name)
    if addresses:
        return addresses[0]
    return None