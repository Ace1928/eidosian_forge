from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_misses(self, x):
    self.has_misses_ = 1
    self.misses_ = x