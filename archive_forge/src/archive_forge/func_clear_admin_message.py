from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_admin_message(self):
    if self.has_admin_message_:
        self.has_admin_message_ = 0
        self.admin_message_ = ''