from netaddr.core import AddrFormatError
from netaddr.ip import IPAddress, IPNetwork

    An generator that yields IPAddress objects from defined by nmap target
    specifications.

    See https://nmap.org/book/man-target-specification.html for details.

    :param nmap_target_spec: one or more nmap IP range target specification.

    :return: an iterator producing IPAddress objects for each IP in the target spec(s).
    