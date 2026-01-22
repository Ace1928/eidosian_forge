from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_delta(self):
    if self.has_delta_:
        self.has_delta_ = 0
        self.delta_ = 1