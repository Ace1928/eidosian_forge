from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_capability(self):
    if self.has_capability_:
        self.has_capability_ = 0
        self.capability_ = ''