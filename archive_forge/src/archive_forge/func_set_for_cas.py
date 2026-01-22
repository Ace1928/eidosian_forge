from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_for_cas(self, x):
    self.has_for_cas_ = 1
    self.for_cas_ = x