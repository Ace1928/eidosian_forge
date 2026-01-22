from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_scope(self):
    if self.has_scope_:
        self.has_scope_ = 0
        self.scope_ = ''