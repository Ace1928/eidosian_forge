from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_request_writer_permission(self):
    if self.has_request_writer_permission_:
        self.has_request_writer_permission_ = 0
        self.request_writer_permission_ = 0