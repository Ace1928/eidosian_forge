from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_expires_in_seconds(self, x):
    self.has_expires_in_seconds_ = 1
    self.expires_in_seconds_ = x