from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_indexed(self):
    if self.has_indexed_:
        self.has_indexed_ = 0
        self.indexed_ = 1