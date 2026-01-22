from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_before(self):
    if self.has_before_:
        self.has_before_ = 0
        self.before_ = 1