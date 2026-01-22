from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_byte_hits(self, x):
    self.has_byte_hits_ = 1
    self.byte_hits_ = x