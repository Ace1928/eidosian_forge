import abc
from os_ken.lib import stringify
Encode a protocol header.

        This method is used only when encoding a packet.

        Encode a protocol header.
        Returns a bytearray which contains the header.

        *payload* is the rest of the packet which will immediately follow
        this header.

        *prev* is a packet_base.PacketBase subclass for the outer protocol
        header.  *prev* is None if the current header is the outer-most.
        For example, *prev* is ipv4 or ipv6 for tcp.serialize.
        