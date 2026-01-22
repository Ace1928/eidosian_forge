from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_computed(self):
    if self.has_computed_:
        self.has_computed_ = 0
        self.computed_ = 0