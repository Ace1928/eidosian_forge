from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_scheduled_time(self):
    if self.has_scheduled_time_:
        self.has_scheduled_time_ = 0
        self.scheduled_time_ = ''