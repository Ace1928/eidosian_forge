from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_multiple(self):
    if self.has_multiple_:
        self.has_multiple_ = 0
        self.multiple_ = 0