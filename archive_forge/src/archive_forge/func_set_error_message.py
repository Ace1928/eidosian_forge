from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_error_message(self, x):
    self.has_error_message_ = 1
    self.error_message_ = x