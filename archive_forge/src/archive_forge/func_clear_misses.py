from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_misses(self):
    if self.has_misses_:
        self.has_misses_ = 0
        self.misses_ = 0