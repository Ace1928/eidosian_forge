from lxml import etree
from .default import DefaultDeviceHandler
from ncclient.operations.third_party.sros.rpc import MdCliRawCommand, Commit
from ncclient.xml_ import BASE_NS_1_0
Set SR OS device handler client capabilities

        Set additional capabilities beyond the default device handler.

        Returns:
            A list of strings representing NETCONF capabilities to be
            sent to the server.
        