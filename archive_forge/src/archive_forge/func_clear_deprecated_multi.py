from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_deprecated_multi(self):
    if self.has_deprecated_multi_:
        self.has_deprecated_multi_ = 0
        self.deprecated_multi_ = 0