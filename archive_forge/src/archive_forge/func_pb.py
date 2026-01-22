import collections
from proto.utils import cached_property
from cloudsdk.google.protobuf.message import Message
@property
def pb(self):
    return self._pb