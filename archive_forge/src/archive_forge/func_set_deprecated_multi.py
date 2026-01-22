from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_deprecated_multi(self, x):
    self.has_deprecated_multi_ = 1
    self.deprecated_multi_ = x