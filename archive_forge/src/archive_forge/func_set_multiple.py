from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_multiple(self, x):
    self.has_multiple_ = 1
    self.multiple_ = x