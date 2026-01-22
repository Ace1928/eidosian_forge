from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util
@property
def data_ele(self):
    """*data* element as an :class:`~xml.etree.ElementTree.Element`"""
    if not self._parsed:
        self.parse()
    return self._data