from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_mode(self):
    if self.has_mode_:
        self.has_mode_ = 0
        self.mode_ = 0