from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_byte_hits(self):
    if self.has_byte_hits_:
        self.has_byte_hits_ = 0
        self.byte_hits_ = 0