from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_internal_message(self):
    if self.has_internal_message_:
        self.has_internal_message_ = 0
        self.internal_message_ = ''